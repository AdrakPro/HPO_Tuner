from dataclasses import dataclass
from enum import Enum, auto

import torch.nn as nn


class OptimizerSchedule(Enum):
    SGD_ONECYCLE = auto()
    SGD_COSINE = auto()
    ADAMW_COSINE = auto()
    ADAMW_ONECYCLE = auto()
    SGD_EXPONENTIAL = auto()
    ADAMW_EXPONENTIAL = auto()

    @property
    def is_adamw(self) -> bool:
        """Returns True if the optimizer is AdamW."""
        return "ADAMW" in self.name

    @property
    def is_onecycle(self) -> bool:
        """Returns True if the scheduler is OneCycleLR."""
        return "ONECYCLE" in self.name

    @property
    def is_cosine(self) -> bool:
        """Returns True if the scheduler is CosineAnnealingLR."""
        return "COSINE" in self.name


class AugmentationIntensity(Enum):
    NONE = auto()
    LIGHT = auto()
    MEDIUM = auto()
    STRONG = auto()


class ActivationFn(Enum):
    RELU = auto()
    LEAKY_RELU = auto()
    GELU = auto()

    def get_layer(self) -> nn.Module:
        if self == ActivationFn.RELU:
            return nn.ReLU(inplace=True)
        elif self == ActivationFn.LEAKY_RELU:
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif self == ActivationFn.GELU:
            return nn.GELU()


@dataclass
class Chromosome:
    width_scale: float
    mixup_alpha: float
    dropout_rate: float
    optimizer_schedule: OptimizerSchedule
    base_lr: float
    aug_intensity: AugmentationIntensity
    weight_decay: float
    batch_size: int
    activation_fn: ActivationFn

    @classmethod
    def from_dict(cls, data: dict):
        """
        Creates a Chromosome object from a dictionary.
        This handles the conversion from primitive types (str) to Enums.
        """
        data_copy = data.copy()
        try:
            data_copy["optimizer_schedule"] = OptimizerSchedule[
                data_copy["optimizer_schedule"].upper()
            ]
            data_copy["aug_intensity"] = AugmentationIntensity[
                data_copy["aug_intensity"].upper()
            ]
            data_copy["activation_fn"] = ActivationFn[
                data_copy["activation_fn"].upper()
            ]
        except KeyError as e:
            raise ValueError(f"Invalid enum value provided in dictionary: {e}")

        return cls(**data_copy)
