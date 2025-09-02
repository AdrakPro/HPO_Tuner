from dataclasses import dataclass
from enum import Enum, auto


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
        Creates a Chromosome object from a dictionary.
        This handles the conversion from primitive types (str) to Enums.
        """
        data_copy = data.copy()
        try:
            data_copy["optimizer_schedule"] = OptimizerSchedule[
                data_copy["optimizer_schedule"]
            ]
            data_copy["aug_intensity"] = AugmentationIntensity[
                data_copy["aug_intensity"]
            ]
        except KeyError as e:
            raise ValueError(f"Invalid enum value provided in dictionary: {e}")

        return cls(**data_copy)
