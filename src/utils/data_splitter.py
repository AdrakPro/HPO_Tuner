import numpy as np
from sklearn.model_selection import StratifiedKFold
from torchvision.datasets import CIFAR10
from typing import List, Tuple


def create_stratified_k_folds(
    data_dir: str, n_splits: int, seed: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates stratified K-folds for the CIFAR-10 dataset.

    This ensures that each fold has a proportional representation of all classes.

    Args:
        data_dir: The directory where the CIFAR-10 dataset is stored.
        n_splits: The number of folds (K) for cross-validation.
        seed: The random seed for shuffling to ensure reproducibility.

    Returns:
        A list of tuples, where each tuple contains (train_indices, test_indices)
        for one fold.
    """
    full_train_set = CIFAR10(root=data_dir, train=True, download=True)
    targets = np.array(full_train_set.targets)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_indices = []
    for train_idx, test_idx in skf.split(np.zeros(len(targets)), targets):
        fold_indices.append((train_idx, test_idx))

    return fold_indices
