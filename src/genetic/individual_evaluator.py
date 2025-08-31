"""
Handles the evaluation of a population of individuals (chromosomes).
"""

from typing import List, Dict, Any, Tuple

from src.genetic.stop_conditions import StopConditions
from src.logger.experiment_logger import logger
from src.model.chromosome import Chromosome
from src.nn.train_and_eval import train_and_eval


class IndividualEvaluator:
    """
    Evaluates a population, respecting the stop conditions.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        training_epochs: int,
        subset_percentage: float = 1.0,
    ):
        """
        Initializes the IndividualEvaluator.

        Args:
            config: The main configuration dictionary.
            training_epochs: The number of epochs to train each individual.
        """
        self.config = config
        self.training_epochs = training_epochs
        self.subset_percentage = subset_percentage

    def evaluate_population(
        self,
        population: List[Dict],
        stop_conditions: StopConditions | None,
        is_final: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """
        Fitness function for the GA. Evaluates each individual and respects stop conditions.

        Args:
            population: A list of dictionaries, where each dictionary represents an individual's hyperparameters.
            stop_conditions: An object that determines when to stop evaluation.
            is_final: Flag for last evaluation

        Returns:
            A tuple containing two lists: fitness_scores (e.g., accuracy) and loss_scores.
        """
        fitness_scores, loss_scores = [], []
        for i, individual_dict in enumerate(population):
            logger.info(
                f"Evaluating Individual {i+1}/{len(population)} ({self.training_epochs} epochs)"
            )
            logger.info(f"Hyperparameters: {individual_dict}")
            try:
                chromosome = Chromosome.from_dict(individual_dict)
                accuracy, loss = train_and_eval(
                    chromosome,
                    self.config,
                    self.training_epochs,
                    self.subset_percentage,
                    is_final,
                )
                logger.info(
                    f"Individual {i+1} -> Accuracy: {accuracy:.4f}, Loss: {loss:.4f}"
                )
                fitness_scores.append(accuracy)
                loss_scores.append(loss)

                # Check if this individual's fitness is high enough to stop the generation early
                if stop_conditions and stop_conditions.should_stop_evaluation(
                    accuracy
                ):
                    remaining_count = len(population) - (i + 1)
                    if remaining_count > 0:
                        logger.info(
                            f"Assigning fitness of 0.0 to the {remaining_count} unevaluated individuals."
                        )
                        fitness_scores.extend([0.0] * remaining_count)
                        loss_scores.extend([float("inf")] * remaining_count)
                    break  # Exit the evaluation loop for this generation

            except Exception as e:
                logger.error(f"Error evaluating individual {i+1}: {e}")
                fitness_scores.append(0.0)
                loss_scores.append(float("inf"))

        return fitness_scores, loss_scores
