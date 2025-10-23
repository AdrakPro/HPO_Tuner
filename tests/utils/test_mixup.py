from unittest.mock import patch

import pytest
import torch

from src.utils.mixup import mixup_data, mixup_criterion, mixup_schedule


@patch("numpy.random.beta", return_value=0.6)
@patch("torch.randperm", return_value=torch.tensor([1, 0]))
def test_mixup_data_with_alpha(mock_randperm, mock_beta):
    """Test mixup_data when alpha > 0, using mocked randomness."""
    x = torch.tensor([[1.0, 1.0], [10.0, 10.0]])
    y = torch.tensor([0, 1])
    alpha = 0.4
    lam = 0.6

    mixed_x, y_a, y_b, returned_lam = mixup_data(
        x, y, alpha=alpha, device="cpu"
    )

    expected_mixed_x = lam * x + (1 - lam) * x[torch.tensor([1, 0]), :]

    assert torch.allclose(mixed_x, expected_mixed_x)
    assert torch.equal(y_a, y)
    assert torch.equal(y_b, y[torch.tensor([1, 0])])
    assert returned_lam == lam


def test_mixup_data_no_alpha():
    """Test mixup_data when alpha is 0, which should result in no mixing."""
    x = torch.tensor([[1.0, 1.0], [10.0, 10.0]])
    y = torch.tensor([0, 1])
    alpha = 0.0

    mixed_x, y_a, y_b, returned_lam = mixup_data(
        x, y, alpha=alpha, device="cpu"
    )

    assert torch.equal(mixed_x, x)
    assert returned_lam == 1.0


def test_mixup_criterion():
    """Test the loss calculation for mixed-up data."""
    mock_criterion = lambda pred, target: pred.sum() + target.sum()
    pred = torch.tensor([10.0, 10.0])
    y_a = torch.tensor([1.0, 1.0])
    y_b = torch.tensor([2.0, 2.0])
    lam = 0.7

    # Loss for y_a = (10+10) + (1+1) = 22
    # Loss for y_b = (10+10) + (2+2) = 24
    expected_loss = (lam * 22) + ((1 - lam) * 24)

    loss = mixup_criterion(mock_criterion, pred, y_a, y_b, lam)

    assert loss == pytest.approx(expected_loss)


@pytest.mark.parametrize(
    "epoch, total_epochs, base_alpha, expected_alpha_factor",
    [
        (0, 100, 0.4, 0.0),  # Start of warmup
        (7, 100, 0.4, 7 / 15),
        # (7 / 15 steps where 15 is the warmup duration)
        (15, 100, 0.4, 1.0),  # End of warmup (if logic uses >=), start of hold
        (50, 100, 0.4, 1.0),  # Middle of hold
        (84, 100, 0.4, 1.0),  # End of hold
        (85, 100, 0.4, 0.0),
        # Start of cooldown (if logic assumes immediate drop)
        (95, 100, 0.4, 0.0),  # Middle of cooldown
    ],
)
def test_mixup_schedule(epoch, total_epochs, base_alpha, expected_alpha_factor):
    """Test the mixup schedule at different phases."""
    expected_alpha = base_alpha * expected_alpha_factor

    result_alpha = mixup_schedule(epoch, total_epochs, base_alpha)

    assert result_alpha == pytest.approx(expected_alpha)
