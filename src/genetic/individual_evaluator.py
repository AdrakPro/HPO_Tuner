"""
Handles the sequential evaluation of a population of individuals on a single device.
"""

import time
from typing import Any, Dict, List

import torch
from rich.progress import Progress, TaskID

from src.genetic.stop_conditions import StopConditions
from src.logger.logger import logger
from src.model.chromosome import Chromosome
from src.model.evaluator_interface import Evaluator
from src.model.parallel import Result
from src.nn.train_and_eval import train_and_eval


class IndividualEvaluator(Evaluator):
    """
    Evaluates a population sequentially in the main process.
    """

    def __init__(
        self,
        neural_config: Dict[str, Any],
        training_epochs: int,
        early_stop_epochs: int,
        subset_percentage: float,
        progress: Progress,
        task_id: TaskID,
    ):
        self.neural_config = neural_config
        self.training_epochs = training_epochs
        self.early_stop_epochs = early_stop_epochs
        self.subset_percentage = subset_percentage
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.progress = progress
        self.task_id = task_id
        logger.info(f"Evaluator initialized on device: {self.device}")

    @property
    def num_workers(self) -> int:
        """The individual evaluator runs on the main thread, so it has 1 worker."""
        return 1

    def evaluate_population(
        self,
        population: List[Dict],
        stop_conditions: StopConditions | None = None,
        is_final: bool = False,
    ) -> List[Result]:
        """
        Fitness function for the GA. Evaluates each individual sequentially.

        Args:
            population: A list of hyperparameter dictionaries.
            stop_conditions: An object that determines when to stop evaluation.
            is_final: Flag for last evaluation.

        Returns:
            A list of Result objects, one per individual in the population.
        """

        results: List[Result] = []
        pop_size = len(population)

        for i, ind in enumerate(population):
            start_time = time.perf_counter()
            logger.info(f"Evaluating Individual {i + 1}/{pop_size}")

            try:
                chromosome = Chromosome.from_dict(ind)

                def epoch_logger(
                    epoch,
                    train_acc,
                    train_loss,
                    test_acc,
                    test_loss,
                    early_stop=False,
                    final_msg=None,
                ):
                    """Logs epoch progress directly to the main logger."""
                    if final_msg:
                        logger.info(f"[Sequential] {final_msg}")
                        return

                    line = f"Epoch {epoch}/{self.training_epochs} | Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"

                    if early_stop:
                        line += " / Early stopping triggered"
                    logger.info(line)

                accuracy, loss = train_and_eval(
                    chromosome=chromosome,
                    neural_config=self.neural_config,
                    epochs=self.training_epochs,
                    early_stop_epochs=self.early_stop_epochs,
                    device=self.device,
                    subset_percentage=self.subset_percentage,
                    is_final=is_final,
                    epoch_callback=epoch_logger,
                )
                status = "SUCCESS"
                error_msg = None

            except Exception as e:
                logger.error(f"Error evaluating individual {i}: {e}")
                accuracy = 0.0
                loss = float("inf")
                status = "FAILURE"
                error_msg = str(e)

            duration = time.perf_counter() - start_time
            self.progress.update(self.task_id, advance=1)

            logger.info(
                f"Individual {i + 1} -> Accuracy: {accuracy:.4f}, Loss: {loss:.4f} | Duration: {duration:.2f}s"
            )

            current_result = Result(
                index=i,
                fitness=accuracy,
                loss=loss,
                status=status,
                log_lines=[],
                duration_seconds=duration,
                error_message=error_msg,
            )
            results.append(current_result)

            if stop_conditions:
                should_stop, reason = stop_conditions.should_stop_evaluation(
                    accuracy
                )
                if should_stop:
                    logger.warning(reason)
                    break

        if len(results) < pop_size:
            for i in range(len(results), pop_size):
                results.append(
                    Result(
                        index=i,
                        fitness=0.0,
                        loss=float("inf"),
                        status="CANCELLED",
                        log_lines=[],
                        duration_seconds=0,
                        error_message="Evaluation stopped early.",
                    )
                )

        return results

    def set_training_epochs(self, epochs: int) -> None:
        """Updates the number of training epochs."""
        self.training_epochs = epochs

    def set_subset_percentage(self, subset_percentage: float) -> None:
        """Updates the subset percentage."""
        self.subset_percentage = subset_percentage

    def cleanup_workers(self):
        """No-op for the individual evaluator."""
        pass
