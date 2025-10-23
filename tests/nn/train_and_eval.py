from unittest.mock import MagicMock, patch, call

import pytest
import torch
import torch.nn as nn

from src.model.chromosome import Chromosome, AugmentationIntensity
from src.nn.train_and_eval import train_epoch, evaluate, train_and_eval
from src.utils.exceptions import NumericalInstabilityError


@pytest.fixture
def mock_components():
    """Provides a dictionary of mocked components for a training loop."""
    model = MagicMock(spec=nn.Module)
    model.parameters.return_value = [nn.Parameter(torch.randn(2, 2))]
    loader = [
        (torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))
    ]  # Mock one batch
    criterion = MagicMock(spec=nn.Module)
    optimizer = MagicMock(spec=torch.optim.Optimizer)
    device = torch.device("cpu")
    return {
        "model": model,
        "loader": loader,
        "criterion": criterion,
        "optimizer": optimizer,
        "device": device,
    }


@pytest.fixture
def base_chromosome():
    """Provides a default Chromosome object."""
    return Chromosome(
        width_scale=1.0,
        mixup_alpha=0.0,
        dropout_rate=0.1,
        optimizer_schedule="SGD_COSINE",
        base_lr=0.01,
        aug_intensity="MEDIUM",
        weight_decay=1e-4,
        batch_size=128,
    )


def test_train_epoch_runs(mock_components):
    """Test that a single training epoch runs without errors."""
    loss, acc = train_epoch(
        model=mock_components["model"],
        loader=mock_components["loader"],
        criterion=mock_components["criterion"],
        optimizer=mock_components["optimizer"],
        device=mock_components["device"],
        lam_alpha=0.0,
    )

    mock_components["optimizer"].zero_grad.assert_called()
    mock_components["model"].assert_called()  # Forward pass
    mock_components["optimizer"].step.assert_called()
    assert isinstance(loss, float)
    assert isinstance(acc, float)


def test_train_epoch_nan_loss_handling(mock_components):
    """Test that the training loop catches NaN loss and raises an error after multiple attempts."""
    mock_components["model"].return_value = torch.tensor([[float("nan")]])
    mock_components["criterion"].return_value = torch.tensor(float("nan"))

    with pytest.raises(
        NumericalInstabilityError, match="NaN loss detected multiple times"
    ):
        train_epoch(
            model=mock_components["model"],
            loader=mock_components["loader"] * 10,
            criterion=mock_components["criterion"],
            optimizer=mock_components["optimizer"],
            device=mock_components["device"],
            lam_alpha=0.0,
        )


def test_evaluate_runs(mock_components):
    """Test that a single evaluation run completes without errors."""
    loss, acc = evaluate(
        model=mock_components["model"],
        loader=mock_components["loader"],
        criterion=mock_components["criterion"],
        device=mock_components["device"],
    )

    mock_components["optimizer"].zero_grad.assert_not_called()
    mock_components["optimizer"].step.assert_not_called()
    assert isinstance(loss, float)
    assert isinstance(acc, float)


@patch("src.nn.train_and_eval.get_network")
@patch("src.nn.train_and_eval.get_optimizer_and_scheduler")
@patch("src.nn.train_and_eval.update_train_augmentation")
@patch("src.nn.train_and_eval.train_epoch", return_value=(0.5, 0.8))
@patch("src.nn.train_and_eval.evaluate")
def test_train_and_eval_early_stopping(
    mock_evaluate,
    mock_train_epoch,
    mock_update_aug,
    mock_get_opt,
    mock_get_net,
    base_chromosome,
    mock_components,
):
    """Test that the main loop stops early if accuracy does not improve."""
    mock_evaluate.side_effect = [
        (0.4, 0.90),  # Epoch 1
        (0.4, 0.89),  # Epoch 2
        (0.4, 0.88),  # Epoch 3
    ]
    early_stop_epochs = 2

    train_and_eval(
        chromosome=base_chromosome,
        neural_config={},
        epochs=10,  # High number of epochs
        early_stop_epochs=early_stop_epochs,
        device=mock_components["device"],
        train_loader=mock_components["loader"],
        test_loader=mock_components["loader"],
    )

    assert mock_train_epoch.call_count == 3
    assert mock_evaluate.call_count == 3


@patch("src.nn.train_and_eval.get_network")
@patch("src.nn.train_and_eval.get_optimizer_and_scheduler")
@patch("src.nn.train_and_eval.update_train_augmentation")
@patch("src.nn.train_and_eval.train_epoch", return_value=(0.5, 0.8))
@patch("src.nn.train_and_eval.evaluate", return_value=(0.4, 0.9))
def test_train_and_eval_augmentation_switching(
    mock_evaluate,
    mock_train_epoch,
    mock_update_aug,
    mock_get_opt,
    mock_get_net,
    base_chromosome,
    mock_components,
):
    """Test that augmentation intensity is switched at the correct epochs."""
    epochs = 20
    light_switch_epoch = int(0.1 * epochs)
    strong_switch_epoch = int(0.4 * epochs)

    train_and_eval(
        chromosome=base_chromosome,
        neural_config={},
        epochs=epochs,
        early_stop_epochs=100,
        device=mock_components["device"],
        train_loader=mock_components["loader"],
        test_loader=mock_components["loader"],
    )

    expected_calls = [
        call(mock_components["loader"], AugmentationIntensity.NONE),
        call(mock_components["loader"], AugmentationIntensity.LIGHT),
        call(mock_components["loader"], base_chromosome.aug_intensity),
    ]
    mock_update_aug.assert_has_calls(expected_calls, any_order=False)
