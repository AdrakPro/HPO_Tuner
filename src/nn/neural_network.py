"""
Core CNN model for CIFAR-10 classification.
Allows user-defined number of convolutional blocks with global config for padding, stride, activation.
"""

from enum import Enum
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional
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

from src.model.chromosome import Chromosome


class ActivationFunction(Enum):
    """Enum for activation functions."""

    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    GELU = "gelu"

    def get_layer(self) -> nn.Module:
        if self == ActivationFunction.RELU:
            return nn.ReLU(inplace=True)
        elif self == ActivationFunction.LEAKY_RELU:
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif self == ActivationFunction.GELU:
            return nn.GELU()


class CNN(nn.Module):
    """
    CNN for CIFAR-10 with user-defined conv blocks.
    """

    def __init__(self, chromosome, neural_config: Dict[str, Any]):
        super().__init__()

        current_channels: int = neural_config["input_shape"][0]
        output_classes: int = neural_config["output_classes"]
        conv_blocks: int = neural_config["conv_blocks"]
        base_filters = int(neural_config["fixed_parameters"]["base_filters"])
        activation = ActivationFunction(
            neural_config["fixed_parameters"]["activation_function"]
        ).get_layer()

        # CNN HYPERPARAMETERS
        conv_stride = 1
        kernel_size = 3
        padding = 1

        # Calculate filters per block
        base = int(base_filters * chromosome.width_scale)
        out_channels = [base * (2**i) for i in range(conv_blocks)]

        layers = []

        for i in range(conv_blocks):
            layers.append(
                nn.Conv2d(
                    current_channels,
                    out_channels[i],
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=conv_stride,
                )
            )

            layers.append(nn.BatchNorm2d(out_channels[i]))
            layers.append(activation)

            if i < conv_blocks - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            current_channels = out_channels[i]

        self.conv_layers = nn.Sequential(*layers)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # TODO: rename fc1 units to fc units
        self.classifier = nn.Sequential(
            nn.Dropout(chromosome.dropout_rate),
            nn.Linear(current_channels, chromosome.fc1_units),
            activation,
            nn.Dropout(chromosome.dropout_rate),
            nn.Linear(chromosome.fc1_units, output_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    # TODO make accessible for rest nonlineraties (for gelu use relu)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                nn.init.constant_(m.bias, 0)


def get_network(chromosome: Chromosome, neural_config: Dict[str, Any]) -> CNN:
    """Factory function to create the CNN model."""
    return CNN(chromosome, neural_config)


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
            max_lr=base_lr,  # OneCycle uses max_lr as peak
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.3,  # 30% of cycle for warmup
            div_factor=25.0,  # initial_lr = max_lr/25
            final_div_factor=10000.0,  # final_lr = max_lr/10000
        )
        return optimizer, scheduler

    warmup_epochs = max(1, int(0.05 * epochs))

    # Main scheduler
    if chromosome.optimizer_schedule.is_cosine:
        main_scheduler = CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs  # Account for warmup
        )
    else:
        main_scheduler = ExponentialLR(optimizer, gamma=EXP_DECAY)

    # Always use warmup for stability
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,  # Start at 10% of target LR
        total_iters=warmup_epochs,
    )

    # Combine warmup + main scheduler
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    return optimizer, scheduler
