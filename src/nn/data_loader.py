"""
Data loader module for CIFAR-10 using PyTorch.
Responsible for downloading, loading, and batching the CIFAR-10 dataset.
"""

import gc
import os
import random
import signal
import sys
from typing import Optional, Union

import numpy as np
import torch.cuda
from torch import set_num_threads, set_num_interop_threads
from torch.utils.data import DataLoader, Subset, Dataset
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
# TODO currently framework only supports CIFAR-10
IMG_SIZE = 32


def dataloader_worker_init_fn():
    """
    Sets the number of threads for a dataloader worker to 1.
    This prevents thread over-subscription when the main worker process
    is already multithreaded, which is crucial for data loading performance.
    """
    num_dl_threads = 1

    set_num_threads(num_dl_threads)
    set_num_interop_threads(num_dl_threads)


class SafeCIFAR10(datasets.CIFAR10):
    """CIFAR-10 that catches corrupt images and returns a dummy tensor."""

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as e:
            print(f"[Warning] Error loading sample {index}: {e}")
            # Return dummy image and label
            return torch.zeros(3, 32, 32), 0


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
                if hasattr(loader, "_iterator"):
                    del loader._iterator
                    loader._iterator = None

                dataset = getattr(loader, "dataset", None)
                if isinstance(dataset, Subset) and hasattr(dataset, "dataset"):
                    dataset.dataset = None

            except Exception as e:
                logger.warning(f"Error during DataLoader cleanup: {e}")

        self.loaders.clear()
        gc.collect()


class DatasetWrapper(Dataset):
    """
    A wrapper to apply a specific transform to a Subset of a Dataset.
    """

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        # Get the data from the underlying dataset (which is a Subset)
        data, target = self.dataset[index]

        # Apply the specific transform for this wrapper
        if self.transform:
            data = self.transform(data)

        return data, target

    def __len__(self):
        return len(self.dataset)


def get_dataset_loaders(
    batch_size: int,
    aug_intensity: AugmentationIntensity,
    is_gpu: bool,
    num_dataloader_workers: int,
    subset_percentage: float = 1.0,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
) -> DataLoaderManager:
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    data_dir = os.environ.get("DATA_DIR", ".")
    model_dir = os.path.join(data_dir, task_id, "model_data")
    os.makedirs(model_dir, exist_ok=True)

    transform_train, transform_test = get_transforms(aug_intensity, img_size)

    train_set_cls = SafeCIFAR10
    test_set_cls = SafeCIFAR10

    # Load datasets
    if train_indices is not None and test_indices is not None:
        base_dataset = train_set_cls(
            root=model_dir, train=True, download=False, transform=None
        )
        train_subset = Subset(base_dataset, train_indices)
        test_subset = Subset(base_dataset, test_indices)

        train_set = DatasetWrapper(train_subset, transform=transform_train)
        test_set = DatasetWrapper(test_subset, transform=transform_test)
    else:
        train_set = train_set_cls(
            root=model_dir,
            train=True,
            download=False,
            transform=transform_train,
        )
        test_set = test_set_cls(
            root=model_dir,
            train=False,
            download=False,
            transform=transform_test,
        )
        if subset_percentage < 1.0:
            num_samples = len(train_set)
            subset_size = max(1, int(num_samples * subset_percentage))
            indices = random.sample(range(len(train_set)), subset_size)
            train_set = Subset(train_set, indices)

    # DataLoader args
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_dataloader_workers,
        "pin_memory": is_gpu,
        "worker_init_fn": dataloader_worker_init_fn,
        "multiprocessing_context": "spawn",
        "persistent_workers": True,
        "prefetch_factor": 4,
    }

    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    return DataLoaderManager(train_loader, test_loader, is_gpu_context=is_gpu)


def get_transforms(
    aug_intensity: AugmentationIntensity,
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Get train and test transforms based on augmentation intensity.
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
