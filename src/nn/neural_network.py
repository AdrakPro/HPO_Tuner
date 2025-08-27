"""
Core CNN model for CIFAR-10 classification.
Allows user-defined number of convolutional blocks with global config for padding, stride, activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.logger.experiment_logger import logger
from src.model.chromosome import Chromosome, OptimizerSchedule


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
        stride: int = config["fixed_parameters"]["stride"]
        padding: int = config["fixed_parameters"]["padding"]

        BASE_FILTERS = 32
        base = int(BASE_FILTERS * chromosome.width_scale)
        out_channels = [base * (2**i) for i in range(conv_blocks)]

        layers = []
        self.activation = getattr(F, activation_function.lower())

        for i in range(conv_blocks):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels[i],
                    kernel_size=3,
                    padding=padding,
                    stride=stride,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels[i]))
            layers.append(ActivationLayer(self.activation))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=stride))
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
) -> tuple[optim.Optimizer, LRScheduler | None]:
    """
    Get optimizer and scheduler based on chromosome.

    Args:
        model: Model to optimize.
        chromosome: Chromosome hyperparameters.
        train_loader: DataLoader for OneCycle scheduler.
        epochs: Number of epochs.

    Returns:
        Tuple of optimizer and scheduler (scheduler can be None).
    """
    SGD_MOMENTUM = 0.9

    if chromosome.optimizer_schedule == OptimizerSchedule.SGD_STEP:
        optimizer = optim.SGD(
            model.parameters(),
            lr=chromosome.base_lr,
            momentum=SGD_MOMENTUM,
            weight_decay=chromosome.weight_decay,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=epochs // 2, gamma=0.1
        )
    elif chromosome.optimizer_schedule == OptimizerSchedule.SGD_COSINE:
        optimizer = optim.SGD(
            model.parameters(),
            lr=chromosome.base_lr,
            momentum=SGD_MOMENTUM,
            weight_decay=chromosome.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    elif chromosome.optimizer_schedule == OptimizerSchedule.ADAMW_COSINE:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=chromosome.base_lr,
            weight_decay=chromosome.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    elif chromosome.optimizer_schedule == OptimizerSchedule.ADAMW_ONECYCLE:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=chromosome.base_lr,
            weight_decay=chromosome.weight_decay,
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=chromosome.base_lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
        )
    else:
        logger.error(
            f"Unknown optimizer schedule: {chromosome.optimizer_schedule}"
        )
        raise ValueError(
            f"Unknown optimizer schedule: {chromosome.optimizer_schedule}"
        )
    return optimizer, scheduler
