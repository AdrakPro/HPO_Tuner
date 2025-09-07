"""
Handles the sequential evaluation of a population of individuals (chromosomes).
"""

from typing import Any, Dict, List, Tuple

import torch

from src.genetic.stop_conditions import StopConditions
from src.logger.logger import logger
from src.model.chromosome import Chromosome
from src.model.evaluator_interface import Evaluator
from src.nn.train_and_eval import train_and_eval


class IndividualEvaluator(Evaluator):
    """
    Evaluates a population sequentially in the main process.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        training_epochs: int,
        early_stop_epochs: int,
        subset_percentage: float,
    ):
        """
        Initializes the IndividualEvaluator.

        Args:
            training_epochs: The number of epochs to train each individual.
            early_stop_epochs: Number of epochs without improvement for early stop.
            subset_percentage: Fraction of the dataset to use.
        """
        self.training_epochs = training_epochs
        self.subset_percentage = subset_percentage
        self.early_stop_epochs = early_stop_epochs
        self.neural_config = config["neural_network_config"]

        eval_mode = config["parallel_config"]["execution"]["evaluation_mode"]

        if eval_mode == "HYBRID":
            logger.warning(
                "Sequential evaluator does not support HYBRID mode. Falling back to CPU."
            )
            self.device = torch.device("cpu")
        elif eval_mode == "GPU" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Sequential evaluator initialized on device: CUDA")
        else:
            if eval_mode == "GPU" in eval_mode:
                logger.warning(
                    "GPU mode requested, but CUDA not available. Falling back to CPU."
                )
            self.device = torch.device("cpu")
            logger.info("Sequential evaluator initialized on device: CPU")

    # TODO: is_final works correctly? how it saves?
    def evaluate_population(
        self,
        population: List[Dict],
        stop_conditions: StopConditions | None,
        is_final: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """
        Fitness function for the GA. Evaluates each individual sequentially.

        Args:
            population: A list of hyperparameter dictionaries.
            stop_conditions: An object that determines when to stop evaluation.
            is_final: Flag for last evaluation.

        Returns:
            A tuple containing (fitness_scores, loss_scores).
        """
        fitness_scores, loss_scores = [], []
        for i, individual_dict in enumerate(population):
            logger.info(
                f"Evaluating Individual {i + 1}/{len(population)} ({self.training_epochs} epochs)"
            )
            logger.info(f"Hyperparameters: {individual_dict}")
            try:
                chromosome = Chromosome.from_dict(individual_dict)
                accuracy, loss = train_and_eval(
                    chromosome,
                    self.neural_config,
                    self.training_epochs,
                    self.early_stop_epochs,
                    self.device,
                    self.subset_percentage,
                    is_final,
                )
                fitness_scores.append(accuracy)
                loss_scores.append(loss)

                logger.info(
                    f"Individual {i + 1} -> Accuracy: {accuracy:.4f}, Loss: {loss:.4f}"
                )

                if stop_conditions:
                    should_stop, reason = (
                        stop_conditions.should_stop_evaluation(accuracy)
                    )

                    if should_stop:
                        logger.warning(reason)
                        remaining_count = len(population) - (i + 1)
                        if remaining_count > 0:
                            fitness_scores.extend([0.0] * remaining_count)
                            loss_scores.extend([float("inf")] * remaining_count)
                        return fitness_scores, loss_scores

            except Exception as e:
                logger.error(f"Error evaluating individual {i + 1}: {e}")
                fitness_scores.append(0.0)
                loss_scores.append(float("inf"))

        return fitness_scores, loss_scores

    def set_training_epochs(self, epochs: int) -> None:
        """Updates the number of training epochs."""
        self.training_epochs = epochs

    def set_subset_percentage(self, subset_percentage: float) -> None:
        """Updates the subset percentage."""
        self.subset_percentage = subset_percentage
