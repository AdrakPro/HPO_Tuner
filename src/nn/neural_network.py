"""
Core CNN model for CIFAR-10 classification.
Allows user-defined number of convolutional blocks with global config for padding, stride, activation.
"""

from enum import Enum
from typing import Any, Callable, Dict, Tuple, Literal

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

        # Weight initialization to prevent gradient explosion and improve training stability
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights using Kaiming initialization based on the network's activation.
        """
        # Map functional activation to PyTorch literal strings
        activation_to_nonlinearity: dict[
            Callable,
            Literal[
                "linear",
                "conv1d",
                "conv2d",
                "conv3d",
                "conv_transpose1d",
                "conv_transpose2d",
                "conv_transpose3d",
                "sigmoid",
                "tanh",
                "relu",
                "leaky_relu",
                "selu",
            ],
        ] = {
            torch.relu: "relu",
            torch.nn.functional.relu: "relu",
            torch.nn.functional.leaky_relu: "leaky_relu",
            torch.nn.functional.gelu: "relu",  # GELU usually uses relu init
        }

        nonlinearity: Literal[
            "linear",
            "conv1d",
            "conv2d",
            "conv3d",
            "conv_transpose1d",
            "conv_transpose2d",
            "conv_transpose3d",
            "sigmoid",
            "tanh",
            "relu",
            "leaky_relu",
            "selu",
        ] = activation_to_nonlinearity.get(self.activation, "relu")

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=nonlinearity
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

    Handles:
      - Adaptive warmup for scaled learning rates
      - Support for different optimizer schedules (SGD, AdamW, OneCycle, Cosine, Exponential)

    Note: base_lr in chromosome should already be scaled and capped in worker.py
    """
    sgd_momentum = 0.9
    exp_decay = 0.95
    weight_decay = chromosome.weight_decay
    base_lr = chromosome.base_lr  # Already scaled and capped in worker.py
    warmup_epochs = max(1, int(0.1 * epochs))

    schedule_type = chromosome.optimizer_schedule

    REFERENCE_BATCH = 256

    # Create optimizer
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

    # OneCycleLR (handles its own warmup)
    if schedule_type.is_onecycle:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=base_lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
        )
        return optimizer, scheduler

    # Setup main scheduler
    if schedule_type.is_cosine:
        main_scheduler = CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs
        )
    else:
        main_scheduler = ExponentialLR(optimizer, gamma=exp_decay)

    # Add warmup for large batch sizes
    if chromosome.batch_size > REFERENCE_BATCH:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,  # Start at 1% of base_lr (already scaled/capped)
            total_iters=warmup_epochs,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        # No warmup needed for small batches
        scheduler = main_scheduler

    return optimizer, scheduler
