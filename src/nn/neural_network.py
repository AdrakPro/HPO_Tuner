"""
Core CNN model for CIFAR-10 classification.
Allows user-defined number of convolutional blocks with global config for padding, stride, activation.
"""

from enum import Enum
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import optim
from torch.optim.lr_scheduler import (
    LRScheduler,
    LinearLR,
    SequentialLR,
    CosineAnnealingLR,
    ExponentialLR,
    OneCycleLR,
)
from torch.utils.data import DataLoader

from src.model.chromosome import Chromosome


class ActivationFunction(Enum):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    GELU = "gelu"


class ActivationLayer(nn.Module):
    """
    Wrapper to use functional activation from torch.nn.functional in nn.Sequential.
    """

    def __init__(self, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn

    def forward(self, x):
        return self.activation_fn(x)


class CNN(nn.Module):
    """
    CNN for CIFAR-10 with user-defined conv blocks.
    """

    def __init__(self, chromosome: Chromosome, config: any) -> None:
        super().__init__()

        conv_blocks: int = config["conv_blocks"]
        in_channels: int = config["input_shape"][0]
        output_classes: int = config["output_classes"]
        activation_function: str = config["fixed_parameters"][
            "activation_function"
        ]
        # Conv layers preserve dimensions
        conv_stride = 1
        # Pooling to downsample
        pool_stride = 2
        padding = 1
        base_filters: int = config["fixed_parameters"]["base_filters"]

        base = int(base_filters * chromosome.width_scale)
        out_channels = [base * (2**i) for i in range(conv_blocks)]

        layers = []
        self.activation = getattr(functional, activation_function.lower())

        for i in range(conv_blocks):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels[i],
                    kernel_size=3,
                    padding=padding,
                    stride=conv_stride,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels[i]))
            layers.append(ActivationLayer(self.activation))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=pool_stride))
            in_channels = out_channels[i]

        self.conv_seq = nn.Sequential(*layers)

        # Dynamically calculate the flatten size for fc1 using a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, *config["input_shape"])
            out = self.conv_seq(dummy)
            fc1_in_features = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(fc1_in_features, chromosome.fc1_units)
        self.fc2 = nn.Linear(chromosome.fc1_units, output_classes)
        self.dropout = nn.Dropout(chromosome.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.conv_seq(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_network(chromosome: Chromosome, config: any) -> CNN:
    return CNN(chromosome, config)


def get_optimizer_and_scheduler(
    model: nn.Module,
    chromosome: Chromosome,
    train_loader: DataLoader,
    epochs: int,
) -> Tuple[optim.Optimizer, LRScheduler | None]:
    """
    Get optimizer and scheduler based on chromosome configuration.
    """
    sgd_momentum = 0.9
    exp_decay = 0.95
    base_lr = chromosome.base_lr
    weight_decay = chromosome.weight_decay
    warmup_epochs = 3
    start_warmup_lr = 0.003

    def make_optimizer(use_adamw: bool) -> optim.Optimizer:
        """Create either SGD or AdamW optimizer."""
        if use_adamw:
            return optim.AdamW(
                model.parameters(), lr=base_lr, weight_decay=weight_decay
            )
        return optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=sgd_momentum,
            weight_decay=weight_decay,
        )

    # TODO: Waiting for fix: https://github.com/pytorch/pytorch/issues/76113
    def apply_warmup(opt: optim.Optimizer, main_scheduler: LRScheduler):
        """Optionally prepend warmup scheduler."""
        if (
            base_lr <= start_warmup_lr
            and chromosome.optimizer_schedule.name
            not in [
                "ADAMW_COSINE",
                "SGD_COSINE",
            ]
        ):
            return main_scheduler
        warmup = LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
        return SequentialLR(
            opt, [warmup, main_scheduler], milestones=[warmup_epochs]
        )

    schedule_type = chromosome.optimizer_schedule.name

    if schedule_type in ("SGD_COSINE", "ADAMW_COSINE"):
        optimizer = make_optimizer("ADAMW" in schedule_type)
        cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        scheduler = apply_warmup(optimizer, cosine)

    elif schedule_type in ("SGD_EXPONENTIAL", "ADAMW_EXPONENTIAL"):
        optimizer = make_optimizer("ADAMW" in schedule_type)
        exp = ExponentialLR(optimizer, gamma=exp_decay)
        scheduler = apply_warmup(optimizer, exp)

    elif schedule_type in ("SGD_ONECYCLE", "ADAMW_ONECYCLE"):
        optimizer = make_optimizer("ADAMW" in schedule_type)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=base_lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
        )

    else:
        raise ValueError(f"Unknown optimizer schedule: {schedule_type}")

    return optimizer, scheduler
