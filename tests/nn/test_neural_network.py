from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.model.chromosome import (
    Chromosome,
    OptimizerSchedule,
    AugmentationIntensity,
)
from src.nn.neural_network import (
    CNN,
    ActivationFunction,
    get_network,
    get_optimizer_and_scheduler,
)


@pytest.fixture
def neural_config():
    """Provides a default neural network configuration."""
    return {
        "input_shape": [3, 32, 32],
        "output_classes": 10,
        "conv_blocks": 3,
        "fixed_parameters": {
            "base_filters": 64,
            "activation_function": "relu",
        },
    }


@pytest.fixture
def base_chromosome():
    """Provides a default Chromosome object."""
    return Chromosome(
        width_scale=1.0,
        mixup_alpha=0.2,
        dropout_rate=0.1,
        optimizer_schedule=OptimizerSchedule.SGD_COSINE,
        base_lr=0.01,
        aug_intensity=AugmentationIntensity.MEDIUM,
        weight_decay=1e-4,
        batch_size=128,
    )


@pytest.fixture
def mock_model():
    """Creates a mock nn.Module with parameters."""
    model = nn.Sequential(nn.Linear(10, 2))
    return model


@pytest.fixture
def mock_train_loader():
    """Creates a mock DataLoader that has a length."""
    loader = MagicMock()
    loader.__len__.return_value = 100  # Mock number of batches
    return loader


def test_activation_function_get_layer():
    """Test that the enum returns the correct nn.Module instance."""
    assert isinstance(ActivationFunction.RELU.get_layer(), nn.ReLU)
    assert isinstance(ActivationFunction.LEAKY_RELU.get_layer(), nn.LeakyReLU)
    assert isinstance(ActivationFunction.GELU.get_layer(), nn.GELU)


class TestCNN:
    """Tests for the CNN model class."""

    def test_cnn_initialization(self, base_chromosome, neural_config):
        """Test if the CNN model is initialized with the correct architecture."""
        model = CNN(base_chromosome, neural_config)

        assert any(isinstance(layer, nn.Dropout) for layer in model.features)

        conv_layers = [
            layer for layer in model.features if isinstance(layer, nn.Conv2d)
        ]
        assert len(conv_layers) == neural_config["conv_blocks"]

    def test_cnn_no_dropout(self, base_chromosome, neural_config):
        """Test that no dropout layers are added if dropout_rate is 0."""
        base_chromosome.dropout_rate = 0.0
        model = CNN(base_chromosome, neural_config)
        assert not any(
            isinstance(layer, nn.Dropout) for layer in model.features
        )

    def test_cnn_forward_pass(self, base_chromosome, neural_config):
        """Test the forward pass with a dummy input tensor."""
        batch_size = 4
        model = CNN(base_chromosome, neural_config)
        dummy_input = torch.randn(batch_size, *neural_config["input_shape"])

        output = model.forward(dummy_input)

        assert output.shape == (batch_size, neural_config["output_classes"])

    @patch("torch.nn.init.kaiming_normal_")
    @patch("torch.nn.init.constant_")
    def test_cnn_weight_initialization(
        self,
        mock_constant_init,
        mock_kaiming_init,
        base_chromosome,
        neural_config,
    ):
        """Test if weight initialization functions are called."""
        _ = CNN(base_chromosome, neural_config)

        assert mock_kaiming_init.called

        assert mock_constant_init.called


def test_get_network_factory(base_chromosome, neural_config):
    """Test the evaluator function for creating the CNN."""
    model = get_network(base_chromosome, neural_config)
    assert isinstance(model, CNN)


@pytest.mark.parametrize(
    "schedule_enum, expected_optimizer, expected_scheduler, expected_inner_scheduler",
    [
        (
            OptimizerSchedule.SGD_COSINE,
            torch.optim.SGD,
            torch.optim.lr_scheduler.SequentialLR,
            torch.optim.lr_scheduler.CosineAnnealingLR,
        ),
        (
            OptimizerSchedule.SGD_EXPONENTIAL,
            torch.optim.SGD,
            torch.optim.lr_scheduler.SequentialLR,
            torch.optim.lr_scheduler.ExponentialLR,
        ),
        (
            OptimizerSchedule.SGD_ONECYCLE,
            torch.optim.SGD,
            torch.optim.lr_scheduler.OneCycleLR,
            None,
        ),
        (
            OptimizerSchedule.ADAMW_COSINE,
            torch.optim.AdamW,
            torch.optim.lr_scheduler.SequentialLR,
            torch.optim.lr_scheduler.CosineAnnealingLR,
        ),
        (
            OptimizerSchedule.ADAMW_EXPONENTIAL,
            torch.optim.AdamW,
            torch.optim.lr_scheduler.SequentialLR,
            torch.optim.lr_scheduler.ExponentialLR,
        ),
        (
            OptimizerSchedule.ADAMW_ONECYCLE,
            torch.optim.AdamW,
            torch.optim.lr_scheduler.OneCycleLR,
            None,
        ),
    ],
)
def test_get_optimizer_and_scheduler(
    base_chromosome,
    mock_model,
    mock_train_loader,
    schedule_enum,
    expected_optimizer,
    expected_scheduler,
    expected_inner_scheduler,
):
    """
    Test all combinations of optimizers and schedulers based on the chromosome.
    """
    base_chromosome.optimizer_schedule = schedule_enum
    epochs = 50

    optimizer, scheduler = get_optimizer_and_scheduler(
        mock_model, base_chromosome, mock_train_loader, epochs
    )

    assert isinstance(optimizer, expected_optimizer)
    assert isinstance(scheduler, expected_scheduler)

    if expected_inner_scheduler:
        assert len(scheduler._schedulers) == 2
        assert isinstance(
            scheduler._schedulers[0], torch.optim.lr_scheduler.LinearLR
        )
        assert isinstance(scheduler._schedulers[1], expected_inner_scheduler)
