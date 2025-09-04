"""
Custom exceptions for the training and evaluation pipeline.
"""


class TrainingException(Exception):
    """Base exception for errors during the model training/evaluation phase."""

    pass


class NumericalInstabilityError(TrainingException):
    """Raised when loss or another metric becomes NaN or infinity."""

    def __init__(
        self, message: str = "Numerical instability detected (NaN or Inf)."
    ):
        self.message = message
        super().__init__(self.message)


class CudaOutOfMemoryError(TrainingException):
    """Raised specifically for CUDA Out-of-Memory errors."""

    def __init__(self, message: str = "CUDA out of memory."):
        self.message = message
        super().__init__(self.message)


class GPUResourceError(Exception):
    """Raised for errors related to GPU availability or initialization."""

    pass
