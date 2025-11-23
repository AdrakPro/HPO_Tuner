import signal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.nn.data_loader import (
    get_dataset_loaders,
    get_transforms,
    SafeCIFAR10,
    DatasetWrapper,
    DataLoaderManager,
    AugmentationIntensity,
)


class TestSafeCIFAR10:
    def test_getitem_corruption_handling(self):
        """Ensure corrupt images return a zero-tensor instead of crashing."""

        dataset = SafeCIFAR10(root=".", download=False)

        with patch("torchvision.datasets.CIFAR10.__getitem__") as mock_super:
            mock_super.side_effect = RuntimeError("Corrupt image file")

            img, label = dataset[0]

            assert torch.is_tensor(img)
            assert torch.sum(img) == 0
            assert label == 0


class TestGetDatasetLoaders:
    @pytest.fixture
    def env_setup(self, monkeypatch):
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
        monkeypatch.setenv("DATA_DIR", "/tmp/test_data")

    def test_loader_initialization(self, env_setup):
        """Test standard initialization."""
        manager = get_dataset_loaders(
            batch_size=16,
            aug_intensity=AugmentationIntensity.LIGHT,
            is_gpu=False,
            num_dataloader_workers=2,
        )

        assert len(manager.loaders) == 2
        assert manager.loaders[0].batch_size == 16

    def test_subset_sampling(self, env_setup, mocker):
        """Verify correct number of samples when subset_percentage < 1.0."""
        mock_subset = mocker.patch(
            "src.nn.data_loader.Subset", side_effect=Subset
        )

        get_dataset_loaders(
            batch_size=16,
            aug_intensity=AugmentationIntensity.NONE,
            is_gpu=False,
            num_dataloader_workers=2,
            subset_percentage=0.1,
        )

        indices = mock_subset.call_args[0][1]
        assert len(indices) == 10

    def test_explicit_indices_wrapper(self, env_setup):
        """Verify DatasetWrapper is used when explicit indices are provided."""
        train_idx = np.array([0, 1, 2])
        test_idx = np.array([3, 4])

        manager = get_dataset_loaders(
            batch_size=16,
            aug_intensity=AugmentationIntensity.NONE,
            is_gpu=False,
            num_dataloader_workers=2,
            train_indices=train_idx,
            test_indices=test_idx,
        )

        assert isinstance(manager.loaders[0].dataset, DatasetWrapper)
        assert len(manager.loaders[0].dataset) == 3


class TestDataLoaderManager:
    def test_cleanup_dataloaders(self):
        loader = MagicMock(spec=DataLoader)
        loader._iterator = "fake_iterator"

        manager = DataLoaderManager(loader)
        manager._cleanup_dataloaders()

        assert getattr(loader, "_iterator", None) is None

    def test_sigint_handling(self, mocker):
        mock_signal = mocker.patch("signal.signal")
        mocker.patch("signal.getsignal")

        manager = DataLoaderManager()
        with manager:
            mock_signal.assert_called_with(
                signal.SIGINT, manager._handle_sigint
            )


class TestTransforms:
    def test_transforms_structure(self):
        train_t, _ = get_transforms(AugmentationIntensity.STRONG)
        types = [type(t) for t in train_t.transforms]
        assert transforms.ColorJitter in types
        assert transforms.RandomErasing in types
