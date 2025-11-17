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
    CNN for CIFAR-10 with user-defined conv blocks.
    """

    def __init__(self, chromosome, neural_config: Dict[str, Any]):
        super().__init__()

        input_channels: int = neural_config["input_shape"][0]
        output_classes: int = neural_config["output_classes"]
        conv_blocks: int = neural_config["conv_blocks"]
        base_filters = int(neural_config["fixed_parameters"]["base_filters"])
        activation = ActivationFunction(
            neural_config["fixed_parameters"]["activation_function"]
        ).get_layer()

        # CNN HYPERPARAMETERS
        convs_per_block = 2
        num_downsample_layers = 2
        num_segments = num_downsample_layers + 1
        downsample_indices = {
            math.floor((i + 1) * conv_blocks / num_segments) - 1
            for i in range(num_downsample_layers)
        }

        layers = []
        in_channels = input_channels
        filter_multiplier = 1

        for i in range(conv_blocks):
            is_downsample_block = i in downsample_indices
            out_channels = int(
                base_filters * filter_multiplier * chromosome.width_scale
            )

            # Convolutional layers for the block
            for _ in range(convs_per_block - 1):
                layers.append(
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=3, padding=1
                    )
                )
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(activation)
                in_channels = out_channels

            # The final layer of the block handles downsampling
            stride = 2 if is_downsample_block else 1
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation)
            in_channels = out_channels

            # Double the filter count for the next stage after downsampling
            if is_downsample_block:
                filter_multiplier *= 2

            # Add dropout after each block
            if chromosome.dropout_rate > 0:
                layers.append(nn.Dropout(chromosome.dropout_rate))

        self.features = nn.Sequential(*layers)

        # Classifier: 1x1 conv -> Global Avg Pooling
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, output_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)

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


class CNN_SIMPLE(nn.Module):
    """
    CNN for CIFAR-10 based on a 3-block VGG-style architecture
    with a 1x1 Conv + Global Average Pooling classifier.
    """

    def __init__(self, chromosome: Chromosome, neural_config: Dict[str, Any]):
        super().__init__()

        input_channels: int = neural_config["input_shape"][0]
        output_classes: int = neural_config["output_classes"]
        base_filters = int(neural_config["fixed_parameters"]["base_filters"])
        activation_name = neural_config["fixed_parameters"]["activation_function"]
        activation = ActivationFunction(activation_name).get_layer()

        # Get hyperparameters from chromosome
        width_scale = chromosome.width_scale
        dropout_rate = chromosome.dropout_rate

        # --- Define Filter Sizes ---
        # We use a 3-block structure with doubling filters
        f1 = int(base_filters * 1 * width_scale)  # e.g., 32
        f2 = int(base_filters * 2 * width_scale)  # e.g., 64
        f3 = int(base_filters * 4 * width_scale)  # e.g., 128

        layers = []
        in_channels = input_channels

        # --- Block 1 (e.g., 32 filters) ---
        layers.extend(
            [
                nn.Conv2d(in_channels, f1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(f1),
                activation,
                nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            ]
        )
        in_channels = f1

        # --- Block 2 (e.g., 64 filters) ---
        layers.extend(
            [
                nn.Conv2d(in_channels, f2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(f2),
                activation,
                nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            ]
        )
        in_channels = f2

        # --- Block 3 (e.g., 128 filters) ---
        layers.extend(
            [
                nn.Conv2d(in_channels, f3, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(f3),
                activation,
                nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
            ]
        )
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
        self._activation_name = activation_name
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)  # Flatten: [B, C, 1, 1] -> [B, C]

    def _initialize_weights(self):
        # Use 'relu' for kaiming init even if using 'gelu' or 'leaky_relu'
        # as it's a standard, robust initialization.
        init_nonlinearity = "relu"

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

class ResidualBlock(nn.Module):
    """
    A basic residual block with two 3x3 convs.
    """
    def __init__(self, in_channels, out_channels, activation: nn.Module, stride=1):
        super().__init__()

        # Main path
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = activation
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, 
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut (skip) connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # We use a 1x1 conv to match the dimensions 
            # if we are downsampling or increasing filters.
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, 
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Main path
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add the shortcut
        out += self.shortcut(x) 
        
        # Final activation
        out = self.act(out)
        return out

class RES_CNN(nn.Module):
    """
    A Light Residual Network for CIFAR-10.
    """
    def __init__(self, chromosome: Chromosome, neural_config: Dict[str, Any]):
        super().__init__()

        input_channels: int = neural_config["input_shape"][0]
        output_classes: int = neural_config["output_classes"]
        base_filters = int(neural_config["fixed_parameters"]["base_filters"])
        activation_name = neural_config["fixed_parameters"]["activation_function"]
        self.activation = ActivationFunction(activation_name).get_layer()

        # Get hyperparameters from chromosome
        width_scale = chromosome.width_scale
        dropout_rate = chromosome.dropout_rate

        # --- Define Filter Sizes ---
        f0 = int(base_filters * width_scale)       # e.g., 64
        f1 = int(base_filters * 2 * width_scale)   # e.g., 128
        f2 = int(base_filters * 4 * width_scale)   # e.g., 256
        f3 = int(base_filters * 8 * width_scale)   # e.g., 512
        
        # --- Network Stages ---
        
        # Stem: A single conv layer to start
        # Input: 32x32
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, f0, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f0),
            self.activation
        )
        
        # Stage 1: 32x32 -> 16x16
        self.stage1 = self._make_stage(f0, f1, stride=2)
        
        # Stage 2: 16x16 -> 8x8
        self.stage2 = self._make_stage(f1, f2, stride=2)
        
        # Stage 3: 8x8 -> 4x4
        self.stage3 = self._make_stage(f2, f3, stride=2)
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # Global Average Pooling
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(f3, output_classes)
        )

        self._initialize_weights() # Use your existing weight init

    def _make_stage(self, in_channels, out_channels, stride):
        # We just use one ResidualBlock per stage for a "light" network
        return ResidualBlock(in_channels, out_channels, self.activation, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # (Your existing weight init function)
        init_nonlinearity = "leaky_relu" if "leaky_relu" in str(self.activation).lower() else "relu"
        
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

class FAV_CNN(nn.Module):
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

        # Get hyperparameters from chromosome
        width_scale = chromosome.width_scale
        dropout_rate = chromosome.dropout_rate

        # --- Define Filter Sizes ---
        f1 = int(base_filters * 1 * width_scale)  # e.g., 32
        f2 = int(base_filters * 2 * width_scale)  # e.g., 64
        f3 = int(base_filters * 4 * width_scale)  # e.g., 128

        feature_dropout_rate = dropout_rate / 2.0

        layers = []
        in_channels = input_channels

        # --- Block 1 (f1 filters) ---
        layers.extend(self._make_block(in_channels, f1, self.activation))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 32x32 -> 16x16
        if dropout_rate > 0:
            layers.append(nn.Dropout(feature_dropout_rate))  # <-- DROPOUT 1
        in_channels = f1

        # --- Block 2 (f2 filters) ---
        layers.extend(self._make_block(in_channels, f2, self.activation))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 16x16 -> 8x8
        if dropout_rate > 0:
            layers.append(nn.Dropout(feature_dropout_rate))  # <-- DROPOUT 2
        in_channels = f2

        # --- Block 3 (f3 filters) ---
        layers.extend(self._make_block(in_channels, f3, self.activation))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 8x8 -> 4x4
        in_channels = f3

        self.features = nn.Sequential(*layers)

        # --- Classifier: Dropout -> 1x1 Conv -> Global Avg Pooling ---
        classifier_layers = []
        if dropout_rate > 0:
            classifier_layers.append(nn.Dropout(dropout_rate))  # <-- DROPOUT 3 (Head)

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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
            # Second Conv Layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)  # Flatten: [B, C, 1, 1] -> [B, C]

    def _initialize_weights(self):
        # Use 'leaky_relu' for kaiming init if using it, else default to 'relu'
        init_nonlinearity = (
            "leaky_relu" if self._activation_name == ActivationFn.LEAKY_RELU.name else "relu"
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
            max_lr=base_lr,  # OneCycle uses max_lr as peak
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.3,  # 30% of cycle for warmup
            div_factor=25.0,  # initial_lr = max_lr/25
            final_div_factor=10000.0,  # final_lr = max_lr/10000
        )
        return optimizer, scheduler

    warmup_epochs = max(1, int(0.05 * epochs))
    main_scheduler_epochs = max(1, epochs - warmup_epochs)

    # Main scheduler
    if chromosome.optimizer_schedule.is_cosine:
        main_scheduler = CosineAnnealingLR(
            optimizer, T_max=main_scheduler_epochs
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
