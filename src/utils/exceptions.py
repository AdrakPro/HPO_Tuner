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
