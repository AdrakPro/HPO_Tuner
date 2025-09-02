"""
Data loader module for CIFAR-10 using PyTorch.
Responsible for downloading, loading, and batching the CIFAR-10 dataset.
"""

import os

import numpy.random as random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from src.model.chromosome import AugmentationIntensity

NUM_WORKERS = os.cpu_count() // 2

# Precomputed statistics, ensuring images are normalized consistently for CIFAR-10
MEANS = (0.4914, 0.4822, 0.4465)
STDS = (0.2023, 0.1994, 0.2010)

# Padded Image Size: (32 + 2*4) x (32 + 2*4) = 40 x 40
# Then it randomly selects a 32x32 window from within the 40x40 area.
# Range [0, 6]
CROP_PADDING = 4

IMG_SIZE = 32

# TODO: What if dataset is imbalanced. We should balance it but we stick to CIFAR-10


class DataLoaderManager:
    """A context manager to ensure DataLoader workers are properly shut down."""

    def __init__(self, *loaders):
        self.loaders = loaders

    def __enter__(self):
        return self.loaders

    def __exit__(self, exc_type, exc_val, exc_tb):
        for loader in self.loaders:
            del loader

        return False


def get_dataset_loaders(
    batch_size: int,
    aug_intensity: AugmentationIntensity,
    is_gpu: bool,
    subset_percentage: float = 1.0,
) -> DataLoaderManager:
    """
    Get CIFAR-10 train/test DataLoaders.

    Returns:
        (train_loader, test_loader): Tuple of DataLoaders.
    """

    data_dir: str = "./model_data"

    transform_train, transform_test = get_transforms(aug_intensity)

    train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    if subset_percentage < 1.0:
        subset_size = int(len(train_set) * subset_percentage)
        indices = random.choice(len(train_set), subset_size, replace=False)
        train_set = Subset(train_set, indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=is_gpu,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=is_gpu,
    )

    return DataLoaderManager(train_loader, test_loader)


def get_transforms(
    aug_intensity: AugmentationIntensity,
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Get train and test transforms based on augmentation intensity.

    Args:
        aug_intensity: AugmentationIntensity Enum.

    Returns:
        Tuple of train and test transforms.
    """
    if aug_intensity == AugmentationIntensity.NONE:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(MEANS, STDS),
            ]
        )
    elif aug_intensity == AugmentationIntensity.LIGHT:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(MEANS, STDS),
            ]
        )
    elif aug_intensity == AugmentationIntensity.MEDIUM:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(IMG_SIZE, padding=CROP_PADDING),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(MEANS, STDS),
            ]
        )
    elif aug_intensity == AugmentationIntensity.STRONG:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(IMG_SIZE, padding=CROP_PADDING),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(MEANS, STDS),
            ]
        )
    else:
        raise ValueError("Invalid AugmentationIntensity")

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEANS, STDS),
        ]
    )
    return train_transform, test_transform
