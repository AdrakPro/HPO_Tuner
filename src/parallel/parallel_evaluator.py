"""
Handles the parallel evaluation of a population of individuals.
"""

import queue
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.multiprocessing as mp

from src.genetic.scheduling_strategy import SchedulingStrategy
from src.genetic.stop_conditions import StopConditions
from src.logger.logger import logger
from src.model.evaluator_interface import Evaluator
from src.model.parallel import Result


# TODO: Enhance DDP Pytorch
class ParallelEvaluator(Evaluator):
    """
    Orchestrates the evaluation of a population across multiple processes.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        training_epochs: int,
        early_stop_epochs: int,
        subset_percentage: float,
        strategy: SchedulingStrategy,
    ):
        self.training_epochs = training_epochs
        self.early_stop_epochs = early_stop_epochs
        self.subset_percentage = subset_percentage
        self.execution_config = config["parallel_config"]["execution"]
        self.neural_network_config = config["neural_network_config"]
        self.strategy = strategy

        self._workers: List[mp.Process] = []
        self._task_queue: mp.Queue | None = None
        self._result_queue: mp.Queue | None = None
        self._shutting_down = False
        self._cleaned_up = False

    def evaluate_population(
        self,
        population: List[Dict],
        stop_conditions: StopConditions | None,
        is_final: bool = False,
    ) -> Tuple[List[float], List[float]]:
        if self._shutting_down:
            logger.warning("Evaluator is shutting down, skipping evaluation")
            return [0.0] * len(population), [float("inf")] * len(population)

        pop_size = len(population)

        ctx = mp.get_context("spawn")
        self._result_queue = ctx.Queue()
        self._task_queue = ctx.Queue()

        # Delegate all task distribution and worker launching to the strategy object.
        self._workers = self.strategy.distribute_and_run(
            population=population,
            task_queue=self._task_queue,
            result_queue=self._result_queue,
            neural_network_config=self.neural_network_config,
            execution_config=self.execution_config,
            training_epochs=self.training_epochs,
            early_stop_epochs=self.early_stop_epochs,
            subset_percentage=self.subset_percentage,
            is_final=is_final,
        )

        if not self._workers:
            logger.error(
                "The selected scheduling strategy failed to launch any workers. Aborting evaluation."
            )
            return [0.0] * len(population), [float("inf")] * len(population)

        # Collect results
        results_list: List[Result] = [None] * pop_size
        finished = 0

        try:
            while finished < pop_size and not self._shutting_down:
                try:
                    result: Result = self._result_queue.get(timeout=3.0)
                except queue.Empty:
                    continue

                results_list[result.index] = result
                finished += 1

                # Flush logs sequentially per individual
                for line in result.log_lines:
                    logger.info(line)

                if stop_conditions:
                    should_stop, reason = (
                        stop_conditions.should_stop_evaluation(result.fitness)
                    )

                    if should_stop:
                        logger.warning(reason)
                        break
        except RuntimeError as e:
            logger.error(
                f"A critical error occurred in the parallel evaluator: {e}"
            )
            self._shutting_down = True
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt detected. Cleaning up workers...")
            self._shutting_down = True

        # If we're shutting down, don't return invalid results
        if self._shutting_down:
            self.cleanup_workers()
            return [0.0] * len(population), [float("inf")] * len(population)

        fitness_scores = [r.fitness if r else 0.0 for r in results_list]
        loss_scores = [r.loss if r else float("inf") for r in results_list]
        return fitness_scores, loss_scores

    def set_training_epochs(self, epochs: int) -> None:
        self.training_epochs = epochs

    def set_subset_percentage(self, subset_percentage: float) -> None:
        """Updates the subset percentage."""
        self.subset_percentage = subset_percentage

    def cleanup_workers(self):
        if self._cleaned_up:
            return

        logger.info("Cleaning up ParallelEvaluator workers...")
        self._shutting_down = True
        self._cleaned_up = True

        # Clear any remaining tasks
        if self._task_queue:
            try:
                while not self._task_queue.empty():
                    self._task_queue.get_nowait()
            except (queue.Empty, FileNotFoundError):
                pass

        # Drain the results queue
        if self._result_queue:
            try:
                while not self._result_queue.empty():
                    self._result_queue.get_nowait()
            except (queue.Empty, FileNotFoundError):
                pass

        # Send sentinel values to workers
        if self._task_queue:
            for _ in range(len(self._workers)):
                try:
                    self._task_queue.put(None, timeout=1)
                except queue.Full:
                    pass

        # Wait for workers to finish
        if self._workers:
            for p in self._workers:
                if p.is_alive():
                    p.join(timeout=10)

                    if p.is_alive():
                        logger.warning(
                            f"Worker {p.pid} did not terminate gracefully, terminating..."
                        )
                        try:
                            p.terminate()
                            # Poll up to 2 seconds total with short sleeps
                            max_wait = 2.0
                            interval = 0.1
                            waited = 0.0

                            while p.is_alive() and waited < max_wait:
                                time.sleep(interval)
                                waited += interval

                            if p.is_alive():
                                logger.error(
                                    f"Worker {p.pid} still alive after terminate(), killing..."
                                )
                                p.kill()
                        except Exception as e:
                            logger.eroror(
                                f"Exception during worker termination: {e}"
                            )
            self._workers = []

        # Clean up queues
        if self._task_queue:
            try:
                self._task_queue.close()
                self._task_queue.join_thread()
            except:
                pass

        if self._result_queue:
            try:
                self._result_queue.close()
                self._result_queue.join_thread()
            except:
                pass

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
