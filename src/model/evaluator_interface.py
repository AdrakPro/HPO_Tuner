"""
Defines the interface for all population evaluation strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from src.genetic.stop_conditions import StopConditions


class Evaluator(ABC):
    """
    An abstract base class that defines the contract for any class that
    evaluates the fitness of a population in the genetic algorithm.
    """

    @abstractmethod
    def evaluate_population(
        self,
        population: List[Dict],
        stop_conditions: StopConditions | None,
        is_final: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """
        Evaluates a given population of individuals.

        Args:
            population: A list of hyperparameter dictionaries.
            stop_conditions: Object to check for early stopping conditions.
            is_final: Flag indicating if this is the final evaluation run.

        Returns:
            A tuple containing two lists: fitness_scores and loss_scores.
        """
        pass

    @abstractmethod
    def set_training_epochs(self, epochs: int) -> None:
        """
        Update the number of training epochs for future evaluations.

        Args:
            epochs: Number of epochs to train each individual.
        """
        pass
