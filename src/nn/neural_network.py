"""
Core CNN model for CIFAR-10 classification.
Allows user-defined number of convolutional blocks with global config for padding, stride, activation.
"""

from enum import Enum
from typing import Any, Callable, Dict, Tuple

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

    def get_fn(self) -> Callable:
        """Get the corresponding torch.nn.functional function."""
        return getattr(functional, self.value)


class ActivationLayer(nn.Module):
    """
    Wrapper to use functional activation from torch.nn.functional in nn.Sequential.
    """

    def __init__(self, activation_fn: Callable[..., torch.Tensor]):
        super().__init__()
        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_fn(x)


class CNN(nn.Module):
    """
    CNN for CIFAR-10 with user-defined conv blocks.
    """

    def __init__(self, chromosome: Chromosome, neural_config: Dict) -> None:
        super().__init__()

        conv_blocks: int = neural_config["conv_blocks"]
        in_channels: int = neural_config["input_shape"][0]
        output_classes: int = neural_config["output_classes"]

        self.activation = ActivationFunction(
            neural_config["fixed_parameters"]["activation_function"]
        ).get_fn()

        # Conv layers preserve dimensions
        conv_stride = 1
        # Pooling to downsample
        pool_stride = 2

        padding = 1
        base_filters: int = neural_config["fixed_parameters"]["base_filters"]

        base = int(base_filters * chromosome.width_scale)
        out_channels = [base * (2**i) for i in range(conv_blocks)]

        layers = []

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
            dummy = torch.zeros(1, *neural_config["input_shape"])
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
    """
    sgd_momentum = 0.9
    exp_decay = 0.95
    base_lr = chromosome.base_lr
    weight_decay = chromosome.weight_decay
    warmup_epochs = 3

    schedule_type = chromosome.optimizer_schedule

    # Optimizer Creation
    if schedule_type.is_adamw:
        optimizer = optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=sgd_momentum,
            weight_decay=weight_decay,
        )

    # Scheduler Creation
    # OneCycleLR includes its own warmup, so it's handled separately
    if schedule_type.is_onecycle:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=base_lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
        )
        return optimizer, scheduler

    # For other schedulers, create the main scheduler first
    if schedule_type.is_cosine:
        main_scheduler = CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs
        )
    else:
        main_scheduler = ExponentialLR(optimizer, gamma=exp_decay)

    # Apply Warmup if needed
    # A warmup is generally not needed if the base learning rate is already very small.
    start_warmup_lr_threshold = 0.003
    if base_lr < start_warmup_lr_threshold:
        return optimizer, main_scheduler

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )

    # Chain the warmup and main schedulers.
    # TODO: Awaiting fix for SequentialLR with some schedulers:
    # https://github.com/pytorch/pytorch/issues/76113
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    return optimizer, scheduler
