from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torchvision.datasets import CIFAR10


def create_stratified_k_folds(
    data_dir: str, n_splits: int, seed: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates stratified K-folds for the CIFAR-10 dataset.

    This ensures that each fold has a proportional representation of all classes.
    """
    full_train_set = CIFAR10(root=data_dir, train=True, download=False)
    targets = np.array(full_train_set.targets)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_indices = []
    for train_idx, test_idx in skf.split(np.zeros(len(targets)), targets):
        fold_indices.append((train_idx, test_idx))

    return fold_indices
