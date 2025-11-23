from unittest.mock import MagicMock, patch, call

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model.chromosome import AugmentationIntensity
from src.nn.train_and_eval import (
    train_epoch,
    evaluate,
    train_and_eval,
    NumericalInstabilityError,
)


class DummyModel(nn.Module):
    """A simple linear model for testing training loops."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def dummy_device():
    return torch.device("cpu")


@pytest.fixture
def dummy_loader():
    """Creates a small DataLoader with random data."""
    inputs = torch.randn(10, 10)
    targets = torch.randint(0, 2, (10,))
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=2)


@pytest.fixture
def mock_chromosome():
    chrom = MagicMock()
    chrom.mixup_alpha = 0.2
    chrom.aug_intensity = AugmentationIntensity.MEDIUM
    return chrom


@pytest.fixture
def mock_dependencies(mocker):
    """Mocks external dependencies to keep tests isolated."""
    mocks = {}
    mocks["logger"] = mocker.patch("src.nn.train_and_eval.logger")
    mocks["update_aug"] = mocker.patch(
        "src.nn.train_and_eval.update_train_augmentation"
    )
    mocks["model_saver"] = mocker.patch("src.nn.train_and_eval.ModelSaver")
    mocks["mixup_data"] = mocker.patch("src.nn.train_and_eval.mixup_data")
    mocks["mixup_crit"] = mocker.patch("src.nn.train_and_eval.mixup_criterion")

    mocks["mixup_data"].side_effect = lambda x, y, alpha, device: (x, y, y, 1.0)
    mocks["mixup_crit"].side_effect = lambda crit, pred, y_a, y_b, lam: crit(
        pred, y_a
    )

    return mocks


class TestTrainEpoch:

    def test_basic_training_step(
        self, dummy_device, dummy_loader, mock_dependencies
    ):
        """Verifies forward/backward pass runs without error."""
        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        loss, acc = train_epoch(
            model,
            dummy_loader,
            criterion,
            optimizer,
            dummy_device,
            lam_alpha=0.0,
        )

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss > 0
        assert model.training

    def test_mixup_logic(self, dummy_device, dummy_loader, mock_dependencies):
        """Verifies mixup is called when lambda > 0."""
        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        train_epoch(
            model,
            dummy_loader,
            criterion,
            optimizer,
            dummy_device,
            lam_alpha=1.0,
        )

        assert mock_dependencies["mixup_data"].called

    def test_nan_loss_error(self, dummy_device, mock_dependencies):
        """Verifies NumericalInstabilityError is raised on persistent NaN loss."""
        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        inputs = torch.randn(14, 10)
        targets = torch.randint(0, 2, (14,))
        dataset = TensorDataset(inputs, targets)
        large_loader = DataLoader(dataset, batch_size=2)

        criterion = MagicMock()
        criterion.side_effect = lambda *args, **kwargs: torch.tensor(
            float("nan")
        )

        mock_dependencies["mixup_crit"].side_effect = lambda c, *args: c(*args)

        with pytest.raises(
            NumericalInstabilityError, match="NaN loss detected"
        ):
            train_epoch(
                model,
                large_loader,
                criterion,
                optimizer,
                dummy_device,
                lam_alpha=0.0,
            )

    def test_scaler_logic(self, dummy_device, dummy_loader, mock_dependencies):
        """Verifies GradScaler methods are called if scaler is provided."""
        model = DummyModel()
        optimizer = MagicMock()
        criterion = nn.CrossEntropyLoss()
        scaler = MagicMock()

        train_epoch(
            model,
            dummy_loader,
            criterion,
            optimizer,
            dummy_device,
            lam_alpha=0.0,
            scaler=scaler,
        )

        assert scaler.scale.called
        assert scaler.step.called
        assert scaler.update.called


class TestEvaluate:

    def test_evaluation_metrics(self, dummy_device, dummy_loader):
        model = DummyModel()
        criterion = nn.CrossEntropyLoss()

        loss, acc = evaluate(model, dummy_loader, criterion, dummy_device)

        assert isinstance(loss, float)
        assert 0.0 <= acc <= 1.0

    def test_empty_loader(self, dummy_device):
        model = DummyModel()
        criterion = nn.CrossEntropyLoss()
        empty_loader = DataLoader(
            TensorDataset(torch.tensor([]), torch.tensor([])), batch_size=1
        )

        loss, acc = evaluate(model, empty_loader, criterion, dummy_device)

        assert loss == float("inf")
        assert acc == 0.0


class TestTrainAndEval:

    @pytest.fixture
    def mock_factory(self, mocker):
        """Mocks get_network and get_optimizer_and_scheduler."""
        m_net = mocker.patch(
            "src.nn.train_and_eval.get_network", return_value=DummyModel()
        )
        m_opt = mocker.patch(
            "src.nn.train_and_eval.get_optimizer_and_scheduler"
        )

        m_opt.return_value = (MagicMock(spec=torch.optim.SGD), None)
        return m_net, m_opt

    def test_empty_train_loader(
        self, mock_chromosome, dummy_device, mock_dependencies
    ):
        """Should return 0.0 fitness immediately."""
        empty_loader = DataLoader([], batch_size=1)

        acc, loss = train_and_eval(
            mock_chromosome, {}, 10, 5, dummy_device, empty_loader, empty_loader
        )

        assert acc == 0.0
        assert loss == float("inf")

        mock_dependencies["logger"].warning.assert_called_with(
            "Train loader is empty. Skipping training and returning 0 fitness."
        )

    def test_augmentation_schedule(
        self,
        mock_chromosome,
        dummy_device,
        dummy_loader,
        mock_factory,
        mock_dependencies,
    ):
        """Verify augmentation intensity changes at specific epochs."""
        mock_chromosome.aug_intensity = AugmentationIntensity.STRONG

        train_and_eval(
            mock_chromosome,
            {},
            epochs=6,
            early_stop_epochs=10,
            device=dummy_device,
            train_loader=dummy_loader,
            test_loader=dummy_loader,
        )

        update_aug = mock_dependencies["update_aug"]

        assert (
            call(dummy_loader, AugmentationIntensity.NONE)
            in update_aug.call_args_list
        )
        assert (
            call(dummy_loader, AugmentationIntensity.LIGHT)
            in update_aug.call_args_list
        )
        assert (
            call(dummy_loader, AugmentationIntensity.STRONG)
            in update_aug.call_args_list
        )

    def test_early_stopping(
        self,
        mock_chromosome,
        dummy_device,
        dummy_loader,
        mock_factory,
        mock_dependencies,
    ):
        """Verify loop breaks if no improvement."""
        with patch("src.nn.train_and_eval.evaluate") as mock_eval:
            mock_eval.side_effect = [
                (0.5, 0.5),  # Epoch 0
                (0.6, 0.4),  # Epoch 1
                (0.7, 0.3),  # Epoch 2 (Stop)
                (0.8, 0.2),
            ]

            with patch(
                "src.nn.train_and_eval.train_epoch", return_value=(0.1, 0.1)
            ):
                train_and_eval(
                    mock_chromosome,
                    {},
                    epochs=10,
                    early_stop_epochs=2,
                    device=dummy_device,
                    train_loader=dummy_loader,
                    test_loader=dummy_loader,
                )

                assert mock_eval.call_count == 3
                mock_dependencies["logger"].info.assert_any_call(
                    "  Stopping early after epoch 3 due to no improvement for 2 epochs."
                )

    def test_model_saving(
        self,
        mock_chromosome,
        dummy_device,
        dummy_loader,
        mock_factory,
        mock_dependencies,
    ):
        """Verify model saver is invoked if is_final=True."""
        with patch("src.nn.train_and_eval.evaluate", return_value=(0.1, 0.9)):
            with patch(
                "src.nn.train_and_eval.train_epoch", return_value=(0.1, 0.1)
            ):
                train_and_eval(
                    mock_chromosome,
                    {},
                    epochs=1,
                    early_stop_epochs=5,
                    device=dummy_device,
                    train_loader=dummy_loader,
                    test_loader=dummy_loader,
                    is_final=True,
                )

                assert mock_dependencies["model_saver"].called
                assert mock_dependencies["model_saver"].return_value.save.called

    def test_ipex_fallback(
        self,
        mock_chromosome,
        dummy_device,
        dummy_loader,
        mock_factory,
        mock_dependencies,
    ):
        """Verify runtime error handling."""
        cpu_device = torch.device("cpu")

        mock_get_opt = mock_factory[1]
        mock_get_opt.side_effect = RuntimeError("Generic Runtime Error")

        with pytest.raises(RuntimeError):
            train_and_eval(
                mock_chromosome,
                {},
                1,
                1,
                cpu_device,
                dummy_loader,
                dummy_loader,
            )
