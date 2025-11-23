import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)


@pytest.fixture(autouse=True)
def mock_logger(mocker):
    """Silences the logger."""
    return mocker.patch("src.logger.logger.logger")


@pytest.fixture(autouse=True)
def mock_cifar10_base(mocker):
    """
    Patches CIFAR10 internals so SafeCIFAR10 can initialize without data.
    """

    def mock_init(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = lambda: None

        self.data = np.zeros((100, 32, 32, 3), dtype=np.uint8)
        self.targets = [0] * 100
        self.classes = ["c" + str(i) for i in range(10)]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    mocker.patch("torchvision.datasets.CIFAR10.__init__", new=mock_init)

    mocker.patch(
        "torchvision.datasets.CIFAR10._check_integrity", return_value=True
    )

    mocker.patch("torchvision.datasets.CIFAR10.__len__", return_value=100)

    mocker.patch(
        "torchvision.datasets.CIFAR10.__getitem__",
        return_value=(torch.randn(3, 32, 32), 0),
    )
