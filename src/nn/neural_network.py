"""
Core CNN model for CIFAR-10 classification.
Allows user-defined number of convolutional blocks with global config for padding, stride, activation.
"""

import math
from enum import Enum, auto
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LinearLR,
    LRScheduler,
    OneCycleLR,
    SequentialLR,
)
from torch.utils.data import DataLoader

from src.model.chromosome import Chromosome, ActivationFn


class CNN(nn.Module):
    """
    CNN for CIFAR-10 with (Conv-Conv-Pool) x 3 blocks.
    """

    def __init__(self, chromosome: Chromosome, neural_config: Dict[str, Any]):
        super().__init__()

        input_channels: int = neural_config["input_shape"][0]
        output_classes: int = neural_config["output_classes"]
        base_filters = int(neural_config["fixed_parameters"]["base_filters"])
        self.activation = chromosome.activation_fn.get_layer()
        self._activation_name = chromosome.activation_fn.name

        width_scale = chromosome.width_scale
        dropout_rate = chromosome.dropout_rate

        f1 = int(base_filters * 1 * width_scale)
        f2 = int(base_filters * 2 * width_scale)
        f3 = int(base_filters * 4 * width_scale)

        feature_dropout_rate = dropout_rate / 2.0

        layers = []
        in_channels = input_channels

        # --- Block 1 (f1 filters) ---
        layers.extend(self._make_block(in_channels, f1, self.activation))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 32x32 -> 16x16
        if dropout_rate > 0:
            layers.append(nn.Dropout(feature_dropout_rate))
        in_channels = f1

        # --- Block 2 (f2 filters) ---
        layers.extend(self._make_block(in_channels, f2, self.activation))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 16x16 -> 8x8
        if dropout_rate > 0:
            layers.append(nn.Dropout(feature_dropout_rate))
        in_channels = f2

        # --- Block 3 (f3 filters) ---
        layers.extend(self._make_block(in_channels, f3, self.activation))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 8x8 -> 4x4
        in_channels = f3

        self.features = nn.Sequential(*layers)

        # --- Classifier: Dropout -> 1x1 Conv -> Global Avg Pooling ---
        classifier_layers = []
        if dropout_rate > 0:
            classifier_layers.append(nn.Dropout(dropout_rate))

        classifier_layers.extend(
            [
                nn.Conv2d(in_channels, output_classes, kernel_size=1),
                nn.AdaptiveAvgPool2d((1, 1)),
            ]
        )
        self.classifier = nn.Sequential(*classifier_layers)

        # Store activation name for weight init
        self._initialize_weights()

    def _make_block(self, in_channels, out_channels, activation):
        """Helper function to create one (Conv-BN-Act) x 2 block."""
        return [
            # First Conv Layer
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            activation,
            # Second Conv Layer
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            activation,
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)  # Flatten: [B, C, 1, 1] -> [B, C]

    def _initialize_weights(self):
        init_nonlinearity = (
            "leaky_relu"
            if self._activation_name == ActivationFn.LEAKY_RELU.name
            else "relu"
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=init_nonlinearity
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=init_nonlinearity
                )
                nn.init.constant_(m.bias, 0)


def get_network(chromosome: Chromosome, neural_config: Dict[str, Any]) -> CNN:
    """Factory function to create the CNN model."""
    return FAV_CNN(chromosome, neural_config)


def get_optimizer_and_scheduler(
    model: nn.Module,
    chromosome: Chromosome,
    train_loader: DataLoader,
    epochs: int,
) -> Tuple[optim.Optimizer, LRScheduler | None]:
    """
    Get optimizer and scheduler based on chromosome configuration.

    Handles:
      - Adaptive warmup for scaled learning rates
      - Support for different optimizer schedules (SGD, AdamW, OneCycle, Cosine, Exponential)

    Note: base_lr in chromosome should already be scaled and capped in worker.py
    """

    # Hyperparameters
    SGD_MOMENTUM = 0.9
    EXP_DECAY = 0.95

    base_lr = chromosome.base_lr

    # Create optimizer
    if chromosome.optimizer_schedule.is_adamw:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=chromosome.weight_decay,
            betas=(0.9, 0.999),
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=SGD_MOMENTUM,
            weight_decay=chromosome.weight_decay,
            nesterov=True,
        )

    # OneCycleLR handles its own warmup
    if chromosome.optimizer_schedule.is_onecycle:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=base_lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=10000.0,
        )
        return optimizer, scheduler

    warmup_epochs = max(1, int(0.05 * epochs))
    main_scheduler_epochs = max(1, epochs - warmup_epochs)

    if chromosome.optimizer_schedule.is_cosine:
        main_scheduler = CosineAnnealingLR(
            optimizer, T_max=main_scheduler_epochs
        )
    else:
        main_scheduler = ExponentialLR(optimizer, gamma=EXP_DECAY)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs,
    )

    # Combine warmup + main scheduler
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    return optimizer, scheduler
