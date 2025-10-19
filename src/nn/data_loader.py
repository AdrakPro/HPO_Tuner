"""
Data loader module for CIFAR-10 using PyTorch.
Responsible for downloading, loading, and batching the CIFAR-10 dataset.
"""

import gc
import random
import signal
import sys
from typing import Optional, Union

import numpy as np
import torch.cuda
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.logger.logger import logger
from src.model.chromosome import AugmentationIntensity

# Precomputed statistics, ensuring images are normalized consistently for CIFAR-10
MEANS = (0.4914, 0.4822, 0.4465)
STDS = (0.2023, 0.1994, 0.2010)

# Padded Image Size: (32 + 2*4) x (32 + 2*4) = 40 x 40
# Then it randomly selects a 32x32 window from within the 40x40 area.
# Range [0, 6]
CROP_PADDING = 4

IMG_SIZE = 32


def get_num_workers(num_workers_from_config: int) -> int:
    """
    Determine the number of workers for DataLoader.
    If the current process is daemonic, return 0.
    Otherwise, return the number provided from the (adjusted) config.
    """
    # try:
    #     if mp.current_process().daemon:
    #         return 0
    # except:
    #     return 0
    return num_workers_from_config

def dataloader_worker_init_fn(worker_id: int):
    """
    Sets the number of threads for a dataloader worker to 1.
    This prevents thread over-subscription when the main worker process
    is already multi-threaded, which is crucial for data loading performance.
    """
    torch.set_num_threads(1)


class DataLoaderManager:
    """A context manager to ensure DataLoader workers are properly shut down."""

    def __init__(self, *loaders, is_gpu_context: bool = False):
        self.loaders = list(loaders)
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)
        self.is_gpu_context = is_gpu_context

    def __enter__(self):
        # Set custom signal handler
        signal.signal(signal.SIGINT, self._handle_sigint)
        return self.loaders

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original signal handler
        signal.signal(signal.SIGINT, self._original_sigint_handler)

        self._cleanup_dataloaders()

        return False

    def _handle_sigint(self, signum, frame):
        """Handle SIGINT by initiating graceful shutdown"""
        sys.exit(1)  # This will trigger __exit__

    def _cleanup_dataloaders(self):
        """Properly cleanup DataLoader workers and resources."""
        for loader in self.loaders:
            try:
                if hasattr(loader, "dataset") and hasattr(
                    loader.dataset, "close"
                ):
                    loader.dataset.close()

                if hasattr(loader, "_iterator"):
                    loader._iterator = None

                if hasattr(loader, "sampler") and hasattr(
                    loader.sampler, "close"
                ):
                    loader.sampler.close()

            except Exception as e:
                logger.warning(f"Error cleaning up DataLoader: {e}")

        self.loaders.clear()
        gc.collect()


def get_dataset_loaders(
    batch_size: int,
    aug_intensity: AugmentationIntensity,
    is_gpu: bool,
    num_dataloader_workers: int,
    subset_percentage: float = 1.0,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
) -> DataLoaderManager:
    """
    Get CIFAR-10 train/test DataLoaders.

    Returns:
        (train_loader, test_loader): Tuple of DataLoaders.
    """

    data_dir = "./model_data"
    transform_train, transform_test = get_transforms(aug_intensity)

    if train_indices is not None and test_indices is not None:
        # Cross-validation fold. The "test set" is a holdout part of the original training set
        full_train_set_with_train_transforms = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        full_train_set_with_test_transforms = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_test
        )

        train_set = Subset(full_train_set_with_train_transforms, train_indices)
        test_set = Subset(full_train_set_with_test_transforms, test_indices)
    else:
        # Use the original full train/test split
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

    num_workers = get_num_workers(num_dataloader_workers)
    enable_persistent_workers = num_workers > 0
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": is_gpu,
        "persistent_workers": enable_persistent_workers,
    }

    if enable_persistent_workers:
        loader_args["worker_init_fn"] = dataloader_worker_init_fn
        loader_args["prefetch_factor"] = 4
        loader_args["multiprocessing_context"] = "spawn"

    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    return DataLoaderManager(train_loader, test_loader, is_gpu_context=is_gpu)


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
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(MEANS, STDS),
            ]
        )
    elif aug_intensity == AugmentationIntensity.MEDIUM:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(IMG_SIZE, padding=CROP_PADDING),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(MEANS, STDS),
            ]
        )
    elif aug_intensity == AugmentationIntensity.STRONG:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(IMG_SIZE, padding=CROP_PADDING),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5
                ),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(MEANS, STDS),
                transforms.RandomErasing(
                    p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)
                ),
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


def update_train_augmentation(
    train_loader: torch.utils.data.DataLoader,
    aug_intensity: Union[AugmentationIntensity, str],
) -> None:
    """
    Dynamically updates the train dataset's augmentation transforms.

    Args:
        train_loader: The DataLoader object for training data.
        aug_intensity: The new augmentation intensity (MEDIUM, STRONG, etc.).
    """
    if isinstance(aug_intensity, str):
        aug_intensity = AugmentationIntensity[aug_intensity.upper()]

    new_train_transform, _ = get_transforms(aug_intensity)

    train_dataset = train_loader.dataset

    if isinstance(train_dataset, torch.utils.data.Subset):
        # Subset wraps the original dataset
        train_dataset.dataset.transform = new_train_transform
    else:
        train_dataset.transform = new_train_transform
