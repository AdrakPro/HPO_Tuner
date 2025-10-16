"""
Handles the parallel evaluation of a population of individuals using a persistent worker pool.
"""

import atexit
import queue
import time
import threading
from collections import deque
import random
from typing import Any, Dict, List, Optional, Callable, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from rich.progress import Progress, TaskID

from src.evaluator.scheduling_strategy import SchedulingStrategy, HybridStrategy
from src.genetic.stop_conditions import StopConditions
from src.logger.logger import logger
from src.model.evaluator_interface import Evaluator
from src.model.parallel import Result, Task
from src.utils.signal_manager import signal_manager

def _task_stealer(
    gpu_queue: mp.Queue,
    cpu_queue: mp.Queue,
    num_gpu_workers: int,
    num_cpu_workers: int,
    stop_event: threading.Event,
):
    """
    Runs in a background thread. Moves surplus tasks from the GPU queue
    to the CPU queue to balance the load, but only if the CPU queue is empty.
    """
    logger.info("Task stealer thread started.")

    # Threshold with a safety buffer. Steal only if there's a significant surplus.
    # This prevents stealing the last few tasks, reserving them for GPUs.
    stealing_threshold = num_gpu_workers + num_cpu_workers

    while not stop_event.is_set():
        try:
            # NEW LOGIC: Steal only if CPU queue is empty AND there's a surplus on GPU queue.
            if cpu_queue.empty() and gpu_queue.qsize() > stealing_threshold:
                task = gpu_queue.get(block=False)
                cpu_queue.put(task)
                logger.info(f"[TaskStealer] Moved task for individual {task.index} to CPU queue.", file_only=True)
        except (queue.Empty, OSError):
            # This is expected if the queue is empty, just continue.
            pass
        except Exception as e:
            logger.error(f"[TaskStealer] Unexpected error: {e}")

        time.sleep(15)
    logger.info("Task stealer thread stopped.")


class ParallelEvaluator(Evaluator):
    """
    Orchestrates evaluation across multiple processes using a persistent worker pool.
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
        self._fitness_goal_met = False
        self._goal_achiever_index = None
        self._pending_tasks: Dict[int, Task] = {}

        self.num_gpu_workers = self.execution_config.get("gpu_workers", 0)
        self.num_cpu_workers = self.execution_config.get("cpu_workers", 0)
        ctx = mp.get_context("spawn")
        self.result_queue: mp.Queue = ctx.Queue(maxsize=2000)

        self.gpu_task_queue: Optional[mp.Queue] = None
        self.cpu_task_queue: Optional[mp.Queue] = None
        self.task_queues: List[mp.Queue] = []
        self._stealer_thread: Optional[threading.Thread] = None
        self._stop_stealer_event: Optional[threading.Event] = None
        
        logger.info("Initializing worker pool...")
        launch_results = self.strategy.launch_workers(
            ctx=ctx,
            result_queue=self.result_queue,
            execution_config=self.execution_config,
            session_log_filename=session_log_filename,
            fixed_batch_size=fixed_batch_size,
        )
        self._workers: List[mp.Process] = launch_results["workers"]

        # Check if the current strategy is the main HybridStrategy (with task stealing)
        if isinstance(self.strategy, HybridStrategy):
            self.gpu_task_queue = launch_results["gpu_task_queue"]
            self.cpu_task_queue = launch_results["cpu_task_queue"]
            self.task_queues = [self.gpu_task_queue, self.cpu_task_queue]
            
            self._stop_stealer_event = threading.Event()
            self._stealer_thread = threading.Thread(
                target=_task_stealer,
                args=(
                    self.gpu_task_queue,
                    self.cpu_task_queue,
                    self.execution_config.get("gpu_workers", 0),
                    self.execution_config.get("cpu_workers", 0),
                    self._stop_stealer_event,
                ),
                daemon=True,
            )
            self._stealer_thread.start()
        else:
            self.task_queues = launch_results["task_queues"]

        if not self._workers:
            raise RuntimeError("The selected scheduling strategy failed to launch any workers.")
        logger.info(f"Worker pool initialized with {len(self._workers)} workers.")

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
        self._fitness_goal_met = False
        self._goal_achiever_index = None

        if self._shutting_down:
            return self._generate_placeholder_results(len(population), "Shutdown in progress.")
        
        tasks = []
        for per_gen_index, ind in enumerate(population):
            tasks.append(Task(
                index=per_gen_index,
                neural_network_config=self.neural_network_config,
                individual_hyperparams=ind,
                training_epochs=self.training_epochs,
                early_stop_epochs=self.early_stop_epochs,
                subset_percentage=self.subset_percentage,
                pop_size=self._current_pop_size,
                is_final=is_final,
                train_indices=self.train_indices,
                test_indices=self.test_indices,
            ))
        
        random.shuffle(tasks)

        # Logika wstępnego wypełniania kolejek
        if isinstance(self.strategy, HybridStrategy) and self.num_cpu_workers > 0:
            # Oblicz, ile zadań wstępnie załadować do CPU
            num_to_preload = min(len(tasks), self.num_cpu_workers)
            
            # Wypełnij kolejkę CPU
            logger.info(f"Pre-loading CPU queue with {num_to_preload} tasks.")
            for i in range(num_to_preload):
                task = tasks[i]
                self._pending_tasks[task.index] = task
                self.cpu_task_queue.put(task)
            
            # Pozostałe zadania umieść w kolejce GPU
            remaining_tasks = tasks[num_to_preload:]
            for task in remaining_tasks:
                self._pending_tasks[task.index] = task
                self.gpu_task_queue.put(task)
        else:
            # Działanie standardowe dla strategii innych niż Hybrid
            target_queue = self.task_queues[0]
            for task in tasks:
                self._pending_tasks[task.index] = task
                target_queue.put(task)
        
        results_for_generation: List[Result] = []
        try:
            while len(results_for_generation) < len(population) and not self._shutting_down:
                try:
                    result: Result = self.result_queue.get(timeout=1.0)
                    self._process_result(result, stop_conditions)
                    results_for_generation.append(result)

                    if self._fitness_goal_met:
                        self._clear_pending_tasks()
                        break
                
                except queue.Empty:
                    if not self._is_any_worker_alive():
                        raise RuntimeError("All workers have died unexpectedly.")
                    continue

        except (RuntimeError, KeyboardInterrupt) as e:
            logger.warning(f"Evaluation interrupted: {e}. Shutting down worker pool...")
            self._shutting_down = True
            raise
        
        if len(results_for_generation) < len(population):
            self._fill_missing_results(results_for_generation, len(population), stop_conditions)

        results_for_generation.sort(key=lambda x: x.index)
        return results_for_generation

    def cleanup_workers(self, timeout_per_worker: float = 30.0) -> None:
        if self._cleaned_up: return
        logger.info("Cleaning up worker pool...")
        self._shutting_down = True
        self._cleaned_up = True

        if self._stealer_thread and self._stop_stealer_event:
            self._stop_stealer_event.set()
            self._stealer_thread.join(timeout=5.0)

        for q in self.task_queues:
            for _ in range(self.num_workers * 2): # Send plenty of sentinels
                try: q.put_nowait(None)
                except (queue.Full, OSError): break
        
        alive_workers = [p for p in self._workers if p.is_alive()]
        for p in alive_workers:
            p.join(timeout=5.0)
        
        still_alive = [p for p in alive_workers if p.is_alive()]
        if still_alive:
            logger.warning(f"Force terminating {len(still_alive)} stuck workers")
            for p in still_alive: self._terminate_process(p)

        self._drain_queues()
        self._close_queues()
        self._workers.clear()

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        signal_manager.unregister_cleanup_handler(self.cleanup_workers)
        logger.info("Worker pool cleanup complete.")
        
    def _drain_queues(self):
        for q in self.task_queues + [self.result_queue]:
            try:
                while not q.empty(): q.get_nowait()
            except (queue.Empty, OSError): pass

    def _close_queues(self):
        for q in self.task_queues + [self.result_queue]:
            try:
                q.close(); q.join_thread()
            except (OSError, ValueError): pass

    def _process_result(self, result: Result, stop_conditions: Optional[StopConditions]):
        if result.index in self._pending_tasks:
            del self._pending_tasks[result.index]
        if self.task_id is not None:
            self.progress.update(self.task_id, advance=1)
        for entry in result.log_lines:
            logger.info(entry[0] if isinstance(entry, tuple) else entry, 
                        file_only=isinstance(entry, tuple) and entry[1] == "file_only")
        if stop_conditions and result.status == "SUCCESS" and result.fitness >= stop_conditions.fitness_goal:
            self._fitness_goal_met = True
            self._goal_achiever_index = result.index
            logger.info(f"Fitness goal {stop_conditions.fitness_goal} met by individual {result.index}. Triggering early stop.")

    def _is_any_worker_alive(self) -> bool:
        return any(p.is_alive() for p in self._workers)

    def set_training_epochs(self, epochs: int): self.training_epochs = epochs
    def set_subset_percentage(self, subset_percentage: float): self.subset_percentage = subset_percentage
    @staticmethod
    def _terminate_process(p: mp.Process):
        try:
            if p.is_alive(): p.terminate(); p.join(2.0)
            if p.is_alive(): p.kill(); p.join(1.0)
        except Exception as e: logger.error(f"Error terminating process {p.pid}: {e}")
    @property
    def num_workers(self) -> int: return len(self._workers)
    def set_task_id(self, task_id: TaskID): self.task_id = task_id
