from unittest.mock import MagicMock, PropertyMock

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import OneCycleLR, SequentialLR

from src.nn.neural_network import CNN, get_network, get_optimizer_and_scheduler


@pytest.fixture
def mock_activation():
    """Mocks the ActivationFn enum/class."""
    mock_fn = MagicMock()
    mock_fn.get_layer.return_value = nn.ReLU()
    mock_fn.name = "RELU"
    return mock_fn


@pytest.fixture
def mock_chromosome(mock_activation):
    """
    Creates a mock Chromosome with configurable properties.
    Default: SGD, ExponentialLR, Standard Width.
    """
    chrom = MagicMock()
    chrom.width_scale = 1.0
    chrom.dropout_rate = 0.5
    chrom.base_lr = 0.1
    chrom.weight_decay = 1e-4
    chrom.activation_fn = mock_activation

    schedule = MagicMock()
    type(schedule).is_adamw = PropertyMock(return_value=False)
    type(schedule).is_onecycle = PropertyMock(return_value=False)
    type(schedule).is_cosine = PropertyMock(return_value=False)
    chrom.optimizer_schedule = schedule

    return chrom


@pytest.fixture
def neural_config():
    """Standard CIFAR-10 config."""
    return {
        "input_shape": (3, 32, 32),
        "output_classes": 10,
        "base_filters": 16,
        "conv_blocks": 3,
    }


@pytest.fixture
def mock_dataloader():
    """Mock dataloader for len() calls in scheduler calculation."""
    loader = MagicMock()
    loader.__len__.return_value = 100  # 100 batches
    return loader


class TestCNNStructure:

    def test_output_shape(self, mock_chromosome, neural_config):
        """Verify the model outputs (Batch_Size, Num_Classes)."""
        model = CNN(mock_chromosome, neural_config)

        x = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 10)

    def test_variable_blocks(self, mock_chromosome, neural_config):
        """Verify layer depth changes with config."""
        neural_config["conv_blocks"] = 2
        model_2_blocks = CNN(mock_chromosome, neural_config)

        neural_config["conv_blocks"] = 4
        model_4_blocks = CNN(mock_chromosome, neural_config)

        assert len(model_4_blocks.features) > len(model_2_blocks.features)

    def test_width_scaling(self, mock_chromosome, neural_config):
        """Verify width_scale affects channel dimensions."""
        mock_chromosome.width_scale = 1.0
        model_std = CNN(mock_chromosome, neural_config)

        mock_chromosome.width_scale = 2.0
        model_wide = CNN(mock_chromosome, neural_config)

        conv1_std = model_std.features[0]
        conv1_wide = model_wide.features[0]

        assert conv1_wide.out_channels == int(conv1_std.out_channels * 2)

    def test_dropout_layers_existence(self, mock_chromosome, neural_config):
        """Verify Dropout is added when rate > 0."""
        mock_chromosome.dropout_rate = 0.5
        model = CNN(mock_chromosome, neural_config)

        has_feature_dropout = any(
            isinstance(m, nn.Dropout) for m in model.features
        )
        assert has_feature_dropout

        has_classifier_dropout = any(
            isinstance(m, nn.Dropout) for m in model.classifier
        )
        assert has_classifier_dropout

    def test_no_dropout_layers(self, mock_chromosome, neural_config):
        """Verify Dropout is REMOVED when rate <= 0."""
        mock_chromosome.dropout_rate = 0.0
        model = CNN(mock_chromosome, neural_config)

        has_feature_dropout = any(
            isinstance(m, nn.Dropout) for m in model.features
        )
        assert not has_feature_dropout

        has_classifier_dropout = any(
            isinstance(m, nn.Dropout) for m in model.classifier
        )
        assert not has_classifier_dropout

    def test_weight_initialization_runs(self, mock_chromosome, neural_config):
        """
        Sanity check that custom initialization runs without error.
        We check if weights are not all zeros (except bias).
        """
        model = CNN(mock_chromosome, neural_config)

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                assert torch.std(m.weight).item() > 0
                if m.bias is not None:
                    assert torch.sum(m.bias).item() == 0


class TestOptimizerScheduler:

    def test_optimizer_sgd_selection(
        self, mock_chromosome, neural_config, mock_dataloader
    ):
        """Test default selection of SGD."""
        type(mock_chromosome.optimizer_schedule).is_adamw = PropertyMock(
            return_value=False
        )

        model = CNN(mock_chromosome, neural_config)
        opt, _ = get_optimizer_and_scheduler(
            model, mock_chromosome, mock_dataloader, epochs=10
        )

        assert isinstance(opt, SGD)
        assert opt.defaults["lr"] == mock_chromosome.base_lr

    def test_optimizer_adamw_selection(
        self, mock_chromosome, neural_config, mock_dataloader
    ):
        """Test selection of AdamW."""
        type(mock_chromosome.optimizer_schedule).is_adamw = PropertyMock(
            return_value=True
        )

        model = CNN(mock_chromosome, neural_config)
        opt, _ = get_optimizer_and_scheduler(
            model, mock_chromosome, mock_dataloader, epochs=10
        )

        assert isinstance(opt, AdamW)

    def test_onecycle_scheduler(
        self, mock_chromosome, neural_config, mock_dataloader
    ):
        """Test OneCycleLR creation."""
        type(mock_chromosome.optimizer_schedule).is_onecycle = PropertyMock(
            return_value=True
        )
        type(mock_chromosome.optimizer_schedule).is_adamw = PropertyMock(
            return_value=True
        )

        model = CNN(mock_chromosome, neural_config)
        opt, sched = get_optimizer_and_scheduler(
            model, mock_chromosome, mock_dataloader, epochs=10
        )

        assert isinstance(sched, OneCycleLR)
        assert sched.total_steps == 1000

    def test_sequential_scheduler_cosine(
        self, mock_chromosome, neural_config, mock_dataloader
    ):
        """Test SequentialLR with Cosine annealing."""
        type(mock_chromosome.optimizer_schedule).is_onecycle = PropertyMock(
            return_value=False
        )
        type(mock_chromosome.optimizer_schedule).is_cosine = PropertyMock(
            return_value=True
        )

        model = CNN(mock_chromosome, neural_config)
        opt, sched = get_optimizer_and_scheduler(
            model, mock_chromosome, mock_dataloader, epochs=100
        )

        assert isinstance(sched, SequentialLR)

        assert len(sched._schedulers) == 2

        assert "LinearLR" in str(type(sched._schedulers[0]))
        assert "CosineAnnealingLR" in str(type(sched._schedulers[1]))

    def test_sequential_scheduler_exponential(
        self, mock_chromosome, neural_config, mock_dataloader
    ):
        """Test SequentialLR with Exponential decay."""
        type(mock_chromosome.optimizer_schedule).is_onecycle = PropertyMock(
            return_value=False
        )
        type(mock_chromosome.optimizer_schedule).is_cosine = PropertyMock(
            return_value=False
        )  # Not cosine -> Exp

        model = CNN(mock_chromosome, neural_config)
        opt, sched = get_optimizer_and_scheduler(
            model, mock_chromosome, mock_dataloader, epochs=100
        )

        assert isinstance(sched, SequentialLR)
        assert "ExponentialLR" in str(type(sched._schedulers[1]))


class TestIntegration:
    def test_get_network_factory(self, mock_chromosome, neural_config):
        """Ensure factory function returns a CNN instance."""
        model = get_network(mock_chromosome, neural_config)
        assert isinstance(model, CNN)
