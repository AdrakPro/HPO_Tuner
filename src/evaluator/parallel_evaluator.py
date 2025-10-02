"""
Handles the parallel evaluation of a population of individuals using a persistent worker pool.
"""

import atexit
import queue
import time
from collections import deque
import random
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
from src.utils.signal_manager import signal_manager


class ParallelEvaluator(Evaluator):
    """
    Orchestrates evaluation across multiple processes using a persistent pool of workers.
    Uses an asynchronous pipeline to ensure high throughput by not waiting
    for slow "straggler" workers.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        training_epochs: int,
        early_stop_epochs: int,
        subset_percentage: float,
        strategy: SchedulingStrategy,
        progress: Progress,
        session_log_filename: str,
        train_indices: Optional[np.ndarray],
        test_indices: Optional[np.ndarray],
        fixed_batch_size: Optional[int] = None,
    ):
        self.training_epochs = training_epochs
        self.early_stop_epochs = early_stop_epochs
        self.subset_percentage = subset_percentage
        self.execution_config = config["parallel_config"]["execution"]
        self.neural_network_config = config["neural_network_config"]
        self.strategy = strategy
        self.train_indices = train_indices
        self.test_indices = test_indices

        self._current_pop_size = 0

        self._shutting_down = False
        self._cleaned_up = False

        self.progress = progress
        self.task_id = None

        ctx = mp.get_context("spawn")
        self.task_queue: mp.Queue = ctx.Queue(maxsize=1000)
        self.result_queue: mp.Queue = ctx.Queue(maxsize=1000)

        logger.info("Initializing worker pool...")
        self._workers: List[mp.Process] = self.strategy.launch_workers(
            ctx=ctx,
            task_queue=self.task_queue,
            result_queue=self.result_queue,
            execution_config=self.execution_config,
            session_log_filename=session_log_filename,
            fixed_batch_size=fixed_batch_size,
        )

        if not self._workers:
            raise RuntimeError(
                "The selected scheduling strategy failed to launch any workers."
            )
        logger.info(
            f"Worker pool initialized with {len(self._workers)} workers."
        )

        self._task_id_counter = 0
        self._pending_tasks: Dict[int, Task] = {}
        self._completed_results: deque[Result] = deque()
        self._population_to_evaluate: deque[tuple[int, Dict]] = deque()

        signal_manager.initialize()
        signal_manager.register_cleanup_handler(self.cleanup_workers)
        atexit.register(self.cleanup_workers)

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
        self._current_pop_size = len(population)

        if self._shutting_down:
            logger.warning("Evaluator is shutting down, skipping evaluation")
            return self._generate_placeholder_results(
                len(population), "Shutdown in progress."
            )

        random.shuffle(population)

        for per_gen_index, ind in enumerate(population):
            self._population_to_evaluate.append((per_gen_index, ind))

        pop_size = len(population)
        results_for_generation: List[Result] = []

        try:
            # 2. Main loop: process until we have enough results for the next generation
            while (
                len(results_for_generation) < pop_size
                and not self._shutting_down
            ):
                self._collect_finished_tasks()

                while (
                    self._completed_results
                    and len(results_for_generation) < pop_size
                ):
                    results_for_generation.append(
                        self._completed_results.popleft()
                    )

                self._submit_new_tasks(is_final)

                if len(results_for_generation) < pop_size:
                    try:
                        result: Result = self.result_queue.get(timeout=1.0)
                        self._process_result(result, stop_conditions)

                        self._completed_results.append(result)

                    except queue.Empty:
                        if not self._is_any_worker_alive():
                            raise RuntimeError(
                                "All workers have died unexpectedly."
                            )
                        continue

        except (RuntimeError, KeyboardInterrupt) as e:
            logger.warning(
                f"Evaluation interrupted: {e}. Shutting down worker pool..."
            )
            self._shutting_down = True
            raise

        while len(results_for_generation) < pop_size:
            results_for_generation.extend(
                self._generate_placeholder_results(
                    1, "Evaluation stopped early."
                )
            )

        return results_for_generation[:pop_size]

    def _submit_new_tasks(self, is_final: bool):
        """Submits tasks from the internal queue until all workers are busy."""
        while (
            self._population_to_evaluate
            and len(self._pending_tasks) < self.num_workers
        ):
            per_gen_index, ind = self._population_to_evaluate.popleft()
            task = Task(
                index=per_gen_index,  # Use per-generation index
                neural_network_config=self.neural_network_config,
                individual_hyperparams=ind,
                training_epochs=self.training_epochs,
                early_stop_epochs=self.early_stop_epochs,
                subset_percentage=self.subset_percentage,
                pop_size=self._current_pop_size,
                is_final=is_final,
                train_indices=self.train_indices,
                test_indices=self.test_indices,
            )
            self._pending_tasks[per_gen_index] = task
            self.task_queue.put(task)

    def _collect_finished_tasks(
        self, stop_conditions: StopConditions | None = None
    ):
        """Non-blocking draining the result queue."""
        try:
            while True:
                result: Result = self.result_queue.get_nowait()
                self._process_result(result, stop_conditions)
                self._completed_results.append(result)
        except queue.Empty:
            pass

    def _process_result(
        self, result: Result, stop_conditions: StopConditions | None
    ):
        """Handles a finished result, including logging and checking stop conditions."""
        if result.index in self._pending_tasks:
            del self._pending_tasks[result.index]

        if self.task_id is not None:
            self.progress.update(self.task_id, advance=1)

        for entry in result.log_lines:
            if isinstance(entry, tuple) and entry[1] == "file_only":
                logger.info(entry[0], file_only=True)
            else:
                logger.info(entry)

        if stop_conditions:
            should_stop, reason = stop_conditions.should_stop_evaluation(
                result.fitness
            )
            if should_stop:
                self._clear_pending_tasks()
                logger.info(reason)

    def _is_any_worker_alive(self) -> bool:
        return any(p.is_alive() for p in self._workers)

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

    def cleanup_workers(self, timeout_per_worker: float = 30.0) -> None:
        if self._cleaned_up:
            return

        logger.info("Cleaning up worker pool...")
        self._shutting_down = True
        self._cleaned_up = True

        # 1. Send sentinels with better error handling
        sentinels_sent = 0
        for _ in self._workers:
            try:
                self.task_queue.put_nowait(None)  # Non-blocking put
                sentinels_sent += 1
            except (queue.Full, OSError, ValueError):
                logger.warning("Queue full, skipping sentinel for one worker")

        logger.info(
            f"Sent {sentinels_sent}/{len(self._workers)} shutdown signals"
        )

        # 2. Wait for graceful termination with progress tracking
        start_time = time.time()
        alive_workers = []

        for i, p in enumerate(self._workers):
            if not p.is_alive():
                continue

            remaining_time = timeout_per_worker - (time.time() - start_time)
            if remaining_time <= 0:
                alive_workers.append(p)
                continue

            p.join(timeout=min(5.0, remaining_time))  # Check frequently
            if p.is_alive():
                alive_workers.append(p)

        # 3. Force termination only if necessary
        if alive_workers:
            logger.warning(
                f"Force terminating {len(alive_workers)} stuck workers"
            )
            for p in alive_workers:
                self._terminate_process(p)

        # 4. Safe queue cleanup
        self._drain_queues()

        # 5. Close queues and clear references
        self._close_queues()
        self._workers.clear()

        # 6. GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        signal_manager.unregister_cleanup_handler(self.cleanup_workers)
        logger.info("Worker pool cleanup complete.")

    @staticmethod
    def _terminate_process(p: mp.Process):
        """Safely terminate a process with escalating force."""
        try:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)

            if p.is_alive():
                logger.error(f"Process {p.pid} still alive, killing...")
                p.kill()
                p.join(timeout=1.0)

            if p.is_alive():
                logger.critical(f"Process {p.pid} cannot be killed!")
            else:
                logger.info(f"Process {p.pid} terminated")

        except Exception as e:
            logger.error(f"Error terminating process {p.pid}: {e}")

    def _drain_queues(self):
        """Safely drain queues without adding items."""
        for q in [self.result_queue, self.task_queue]:
            try:
                drained_count = 0
                while not q.empty():
                    q.get_nowait()
                    drained_count += 1
                if drained_count > 0:
                    logger.info(f"Drained {drained_count} items from queue")
            except (queue.Empty, OSError, ValueError):
                pass

    def _close_queues(self):
        """Safely close queues."""
        for q in [self.task_queue, self.result_queue]:
            try:
                q.close()
            except (OSError, ValueError):
                pass

    @property
    def num_workers(self) -> int:
        return len(self._workers)

    def set_task_id(self, task_id: TaskID):
        self.task_id = task_id
