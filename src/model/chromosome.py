from dataclasses import dataclass
from enum import Enum, auto


class OptimizerSchedule(Enum):
    SGD_ONECYCLE = auto()
    SGD_COSINE = auto()
    ADAMW_COSINE = auto()
    ADAMW_ONECYCLE = auto()
    SGD_EXPONENTIAL = auto()
    ADAMW_EXPONENTIAL = auto()


class AugmentationIntensity(Enum):
    NONE = auto()
    LIGHT = auto()
    MEDIUM = auto()
    STRONG = auto()


@dataclass
class Chromosome:
    width_scale: float
    fc1_units: int
    dropout_rate: float
    optimizer_schedule: OptimizerSchedule
    base_lr: float
    aug_intensity: AugmentationIntensity
    weight_decay: float
    batch_size: int

    @classmethod
    def from_dict(cls, data: dict):
        """
        Creates a Chromosome object from a dictionary provided by the GA.
        This handles the conversion from primitive types (str, int) to Enums.
        """
        data_copy = data.copy()

        data_copy["optimizer_schedule"] = OptimizerSchedule[
            data_copy["optimizer_schedule"]
        ]
        data_copy["aug_intensity"] = AugmentationIntensity[
            data_copy["aug_intensity"]
        ]

        return cls(**data_copy)
