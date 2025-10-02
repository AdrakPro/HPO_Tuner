from abc import ABC, abstractmethod
from typing import Dict, List

from rich.progress import TaskID

from src.genetic.stop_conditions import StopConditions
from src.model.parallel import Result


class Evaluator(ABC):
    """
    Abstract base class for an evaluator. Defines the interface for evaluating
    a population and managing evaluation resources.
    It is designed to be used as a context manager.
    """

    def __enter__(self):
        """Allows the evaluator to be used as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures cleanup is called when the context is exited."""
        self.cleanup_workers()

    @property
    @abstractmethod
    def num_workers(self) -> int:
        """Returns the number of active workers used by the evaluator."""
        pass

    @abstractmethod
    def evaluate_population(
        self,
        population: List[Dict],
        stop_conditions: StopConditions | None,
        is_final: bool = False,
    ) -> List[Result]:
        """Evaluates every individual in the population."""
        pass

    @abstractmethod
    def set_training_epochs(self, epochs: int) -> None:
        """Updates the number of training epochs for subsequent evaluations."""
        pass

    @abstractmethod
    def set_task_id(self, task_id: TaskID) -> None:
        """Updates the progress bar task id for subsequent evaluations."""
        pass

    @abstractmethod
    def set_subset_percentage(self, subset_percentage: float) -> None:
        """Updates the data subset percentage for subsequent evaluations."""
        pass

    @abstractmethod
    def cleanup_workers(self) -> None:
        """Cleans up any persistent resources, like worker processes."""
        pass
