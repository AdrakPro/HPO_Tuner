from unittest.mock import MagicMock, patch

import pytest
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.model.chromosome import AugmentationIntensity
from src.nn.data_loader import (
    DataLoaderManager,
    get_dataset_loaders,
    get_transforms,
    update_train_augmentation,
)


@pytest.fixture(autouse=True)
def mock_cifar_dataset(monkeypatch):
    """Mocks torchvision.datasets.CIFAR10 to avoid actual downloads."""
    mock_dataset = MagicMock(spec=datasets.CIFAR10)

    mock_dataset.__len__.return_value = 100

    mock_dataset.return_value = mock_dataset
    monkeypatch.setattr("src.nn.data_loader.datasets.CIFAR10", mock_dataset)
    return mock_dataset


@pytest.mark.parametrize(
    "intensity, expected_transforms",
    [
        (
            AugmentationIntensity.NONE,
            [transforms.ToTensor, transforms.Normalize],
        ),
        (
            AugmentationIntensity.LIGHT,
            [
                transforms.RandomHorizontalFlip,
                transforms.ToTensor,
                transforms.Normalize,
            ],
        ),
        (
            AugmentationIntensity.MEDIUM,
            [
                transforms.RandomCrop,
                transforms.RandomHorizontalFlip,
                transforms.ToTensor,
                transforms.Normalize,
            ],
        ),
        (
            AugmentationIntensity.STRONG,
            [
                transforms.RandomCrop,
                transforms.RandomHorizontalFlip,
                transforms.RandomAffine,
                transforms.ColorJitter,
                transforms.ToTensor,
                transforms.Normalize,
                transforms.RandomErasing,
            ],
        ),
    ],
)
def test_get_transforms(intensity, expected_transforms):
    """Test that get_transforms returns the correct set of transforms for each intensity."""
    train_transform, test_transform = get_transforms(intensity)

    train_transform_types = [type(t) for t in train_transform.transforms]
    assert train_transform_types == expected_transforms

    test_transform_types = [type(t) for t in test_transform.transforms]
    assert test_transform_types == [transforms.ToTensor, transforms.Normalize]


def test_update_train_augmentation():
    """Test that the augmentation of a train loader's dataset can be updated dynamically."""
    mock_dataset = MagicMock(spec=Dataset)
    mock_loader = MagicMock(spec=DataLoader)
    mock_loader.dataset = mock_dataset

    update_train_augmentation(mock_loader, AugmentationIntensity.STRONG)

    assert isinstance(
        mock_dataset.transform.transforms[-1], transforms.RandomErasing
    )


def test_get_dataset_loaders_returns_manager(mock_cifar_dataset):
    """Test that the evaluator function returns a DataLoaderManager."""
    manager = get_dataset_loaders(
        batch_size=32,
        aug_intensity=AugmentationIntensity.MEDIUM,
        is_gpu=False,
        num_dataloader_workers=0,
    )
    assert isinstance(manager, DataLoaderManager)
    assert len(manager.loaders) == 2
    assert isinstance(manager.loaders[0], DataLoader)
    assert isinstance(manager.loaders[1], DataLoader)


@patch("src.nn.data_loader.signal")
def test_dataloader_manager_signal_handling(mock_signal):
    """Test that the DataLoaderManager correctly sets and restores signal handlers."""
    manager = DataLoaderManager(MagicMock())

    with manager:
        mock_signal.signal.assert_called_with(
            mock_signal.SIGINT, manager._handle_sigint
        )

    mock_signal.signal.assert_called_with(
        mock_signal.SIGINT, manager._original_sigint_handler
    )
