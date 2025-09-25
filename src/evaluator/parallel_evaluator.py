"""
Handles the parallel evaluation of a population of individuals using a persistent worker pool.
"""

import queue
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
from rich.progress import Progress, TaskID

from src.evaluator.scheduling_strategy import SchedulingStrategy
from src.genetic.stop_conditions import StopConditions
from src.logger.logger import logger
from src.model.evaluator_interface import Evaluator
from src.model.parallel import Result, Task


class ParallelEvaluator(Evaluator):
    """
    Orchestrates evaluation across multiple processes using a persistent pool of workers.
    This class is a context manager to ensure proper startup and cleanup.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        training_epochs: int,
        early_stop_epochs: int,
        subset_percentage: float,
        strategy: SchedulingStrategy,
        progress: Progress,
        task_id: TaskID,
        session_log_filename: str,
        train_indices: Optional[np.ndarray],
        test_indices: Optional[np.ndarray],
    ):
        self.training_epochs = training_epochs
        self.early_stop_epochs = early_stop_epochs
        self.subset_percentage = subset_percentage
        self.execution_config = config["parallel_config"]["execution"]
        self.neural_network_config = config["neural_network_config"]
        self.strategy = strategy
        self.train_indices = train_indices
        self.test_indices = test_indices

        self._shutting_down = False
        self._cleaned_up = False

        self.progress = progress
        self.task_id = task_id

        ctx = mp.get_context("spawn")
        self.task_queue: mp.Queue = ctx.Queue()
        self.result_queue: mp.Queue = ctx.Queue()

        logger.info("Initializing worker pool...")
        self._workers: List[mp.Process] = self.strategy.launch_workers(
            ctx=ctx,
            task_queue=self.task_queue,
            result_queue=self.result_queue,
            execution_config=self.execution_config,
            session_log_filename=session_log_filename,
        )

        if not self._workers:
            raise RuntimeError(
                "The selected scheduling strategy failed to launch any workers."
            )
        logger.info(
            f"Worker pool initialized with {len(self._workers)} workers."
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_workers()

    def evaluate_population(
        self,
        population: List[Dict],
        stop_conditions: StopConditions | None,
        is_final: bool = False,
    ) -> List[Result]:
        if self._shutting_down:
            logger.warning("Evaluator is shutting down, skipping evaluation")
            return self._generate_placeholder_results(
                len(population), "Shutdown in progress."
            )

        pop_size = len(population)
        for i, ind in enumerate(population):
            self.task_queue.put(
                Task(
                    index=i,
                    neural_network_config=self.neural_network_config,
                    individual_hyperparams=ind,
                    training_epochs=self.training_epochs,
                    early_stop_epochs=self.early_stop_epochs,
                    subset_percentage=self.subset_percentage,
                    pop_size=pop_size,
                    is_final=is_final,
                    train_indices=self.train_indices,
                    test_indices=self.test_indices,
                )
            )

        results_list: List[Result] = [None] * pop_size
        finished = 0

        try:
            while finished < pop_size and not self._shutting_down:
                try:
                    result: Result = self.result_queue.get(timeout=1.0)
                    results_list[result.index] = result
                    finished += 1

                    self.progress.update(self.task_id, advance=1)

                    for entry in result.log_lines:
                        if isinstance(entry, tuple):
                            line, mode = entry
                            if mode == "file_only":
                                logger.info(line, file_only=True)
                        else:
                            logger.info(entry)

                    if stop_conditions:
                        should_stop, reason = (
                            stop_conditions.should_stop_evaluation(
                                result.fitness
                            )
                        )

                        if should_stop:
                            self._clear_pending_tasks()
                            logger.warning(reason)
                            break
                except queue.Empty:
                    continue
        except (RuntimeError, KeyboardInterrupt):
            logger.warning(f"Shutting down worker pool...")
            self._shutting_down = True
            raise

        if finished < pop_size:
            for i in range(pop_size):
                if results_list[i] is None:
                    results_list[i] = Result(
                        index=i,
                        fitness=0.0,
                        loss=float("inf"),
                        status="CANCELLED",
                        log_lines=[],
                        duration_seconds=0,
                        error_message="Evaluation stopped early.",
                    )
        return results_list

    @staticmethod
    def _generate_placeholder_results(
        num_results: int, reason: str
    ) -> List[Result]:
        return [
            Result(
                index=i,
                fitness=0.0,
                loss=float("inf"),
                status="FAILURE",
                log_lines=[],
                duration_seconds=0,
                error_message=reason,
            )
            for i in range(num_results)
        ]

    def _clear_pending_tasks(self):
        logger.info(
            "Early stop triggered. Draining task queue to unblock workers."
        )
        try:
            while not self.task_queue.empty():
                self.task_queue.get_nowait()
        except queue.Empty:
            pass

    def set_training_epochs(self, epochs: int) -> None:
        """Updates the training epochs."""
        self.training_epochs = epochs

    def set_subset_percentage(self, subset_percentage: float) -> None:
        """Updates the subset percentage."""
        self.subset_percentage = subset_percentage

    def cleanup_workers(self, timeout_per_worker: float = 60.0) -> None:
        if self._cleaned_up:
            return

        logger.info("Cleaning up worker pool...")
        self._shutting_down = True
        self._cleaned_up = True

        # 1. Send sentinel values to workers to signal shutdown
        for _ in self._workers:
            try:
                self.task_queue.put(None, timeout=1.0)
            except (queue.Full, OSError, ValueError):
                logger.warning(
                    "Could not send sentinel to worker queue, skipping."
                )
                break

        # 2. Wait for workers to finish gracefully
        for p in self._workers:
            if not p.is_alive():
                continue

            p.join(timeout=timeout_per_worker)
            if p.is_alive():
                try:
                    logger.warning(
                        f"Worker {p.pid} did not exit, terminating..."
                    )
                    p.terminate()
                    p.join(timeout=2.0)
                    if p.is_alive():
                        logger.error(
                            f"Worker {p.pid} still alive after terminate(), killing..."
                        )
                        p.kill()
                except Exception as e:
                    logger.error(f"Error terminating worker {p.pid}: {e}")

        # 3. Drain remaining results safely
        for q in [self.result_queue, self.task_queue]:
            try:
                while not q.empty():
                    item = q.get_nowait()
                    if item is None:
                        fake_result = Result(
                            index=-1,
                            fitness=0.0,
                            loss=float("inf"),
                            status="FAILURE",
                            log_lines=[],
                            duration_seconds=0,
                            error_message="Worker returned None",
                        )
                        q.put(fake_result)
            except (queue.Empty, OSError, ValueError):
                pass

        # 4. Close queues
        for q in [self.task_queue, self.result_queue]:
            try:
                q.close()
                q.join_thread()
            except (OSError, ValueError):
                pass

        # 5. Free GPU memory if applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._workers = []
        logger.info("Worker pool cleanup complete.")

    @property
    def num_workers(self) -> int:
        return len(self._workers)
