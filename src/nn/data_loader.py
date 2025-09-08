"""
Data loader module for CIFAR-10 using PyTorch.
Responsible for downloading, loading, and batching the CIFAR-10 dataset.
"""
import os
import signal
import sys
import random

import torch.multiprocessing as mp
import torch.cuda
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.model.chromosome import AugmentationIntensity

# Precomputed statistics, ensuring images are normalized consistently for CIFAR-10
MEANS = (0.4914, 0.4822, 0.4465)
STDS = (0.2023, 0.1994, 0.2010)

# Padded Image Size: (32 + 2*4) x (32 + 2*4) = 40 x 40
# Then it randomly selects a 32x32 window from within the 40x40 area.
# Range [0, 6]
CROP_PADDING = 4

IMG_SIZE = 32


# TODO: integrate with config x cores
def get_num_workers():
    """
    Determine the number of workers for DataLoader.
    If the current process is daemonic, return 0 because daemonic processes cannot have children.
    Otherwise, return a reasonable number of workers.
    """
    try:
        if mp.current_process().daemon:
            return 0
    except:
        return 0
    return min(4, os.cpu_count() // 2)


class DataLoaderManager:
    """A context manager to ensure DataLoader workers are properly shut down."""

    def __init__(self, *loaders):
        self.loaders = loaders
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)

    def __enter__(self):
        # Set custom signal handler
        signal.signal(signal.SIGINT, self._handle_sigint)
        return self.loaders

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original signal handler
        signal.signal(signal.SIGINT, self._original_sigint_handler)

        # Clean up DataLoaders and their workers
        for loader in self.loaders:
            if hasattr(loader, "_iterator") and loader._iterator is not None:
                try:
                    loader._iterator._shutdown_workers()
                except AttributeError:
                    pass
            del loader

        # Clean up GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return False

    def _handle_sigint(self, signum, frame):
        """Handle SIGINT by initiating graceful shutdown"""
        sys.exit(1)  # This will trigger __exit__


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

    data_dir = "./model_data"

    transform_train, transform_test = get_transforms(aug_intensity)

    train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    if subset_percentage < 1.0:
        subset_size = int(len(train_set) * subset_percentage)
        indices = random.sample(range(len(train_set)), subset_size)
        train_set = Subset(train_set, indices)

    # Use dynamic worker count based on process type
    num_workers = get_num_workers()
    enable_persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=is_gpu,
        persistent_workers=enable_persistent_workers,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=is_gpu,
        persistent_workers=enable_persistent_workers,
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
