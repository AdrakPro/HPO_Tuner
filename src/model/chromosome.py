from dataclasses import dataclass
from enum import Enum, auto


class OptimizerSchedule(Enum):
    SGD_STEP = auto()
    SGD_COSINE = auto()
    ADAMW_COSINE = auto()
    ADAMW_ONECYCLE = auto()


class AugmentationIntensity(Enum):
    NONE = auto()
    LIGHT = auto()
    MEDIUM = auto()
    STRONG = auto()


class BatchSize(Enum):
    B32 = 32
    B64 = 64
    B128 = 128
    B256 = 256


@dataclass
class Chromosome:
    width_scale: float
    fc1_units: int
    dropout_rate: float
    optimizer_schedule: OptimizerSchedule
    base_lr: float
    aug_intensity: AugmentationIntensity
    weight_decay: float
    batch_size: BatchSize
