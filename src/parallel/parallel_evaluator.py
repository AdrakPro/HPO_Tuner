"""
Handles the parallel evaluation of a population of individuals.
"""

import os
import queue
import signal
import sys
import time
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.multiprocessing as mp

from src.genetic.stop_conditions import StopConditions
from src.logger.logger import logger
from src.model.chromosome import Chromosome
from src.model.evaluator_interface import Evaluator
from src.model.parallel import Result, Task, WorkerConfig
from src.nn.train_and_eval import train_and_eval
from src.utils.exceptions import CudaOutOfMemoryError, NumericalInstabilityError


def init_device(device_id: Union[str, int]) -> torch.device:
    """
    Initialize torch device for a worker.
    """
    if isinstance(device_id, int):
        # Set CUDA device after process spawn
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        if not torch.cuda.is_available():
            logger.error(f"CUDA not available for GPU worker {device_id}.")
            return torch.device("cpu")
        else:
            return torch.device("cuda")
    else:
        return torch.device("cpu")


def worker_main(worker_config: WorkerConfig) -> None:
    """
    Top-level worker function so it can be pickled by multiprocessing.
    """
    # Ignore SIGINT during imports to prevent KeyboardInterrupt during initialization
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Set number of threads to avoid CPU oversubscription
    torch.set_num_threads(worker_config.num_threads)
    device = init_device(worker_config.device)
    device_name = (
        f"GPU-{worker_config.device}"
        if isinstance(worker_config.device, int)
        else "CPU"
    )

    # Restore signal handler after imports are complete
    signal.signal(signal.SIGINT, original_sigint_handler)

    # Set up our own signal handler
    def sigint_handler(signum, frame):
        logger.info(
            f"Worker {worker_config.worker_id} received SIGINT, shutting down..."
        )
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    while True:
        try:
            task: Task = worker_config.task_queue.get(timeout=1)
            if task is None:  # Sentinel to stop
                break

            log_buffer: List[str] = [
                f"[Worker-{worker_config.worker_id} / {device_name}] Evaluating Individual {task.index}/{task.pop_size} ({task.training_epochs} epochs)",
                f"[Worker-{worker_config.worker_id} / {device_name}] Hyperparameters: {task.individual_hyperparams}",
            ]

            try:
                chromosome = Chromosome.from_dict(task.individual_hyperparams)

                def epoch_logger(
                    epoch,
                    train_acc,
                    train_loss,
                    test_acc,
                    test_loss,
                    early_stop=False,
                    final_msg=None,
                ):
                    if final_msg:
                        log_buffer.append(
                            f"[Worker-{worker_config.worker_id} / {device_name}] {final_msg}"
                        )
                        return
                    line = f"[Worker-{worker_config.worker_id} / {device_name}] Epoch {epoch}/{task.training_epochs} | Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
                    if early_stop:
                        line += " / Early stopping triggered"
                    log_buffer.append(line)

                accuracy, loss = train_and_eval(
                    chromosome=chromosome,
                    neural_config=task.neural_network_config,
                    epochs=task.training_epochs,
                    early_stop_epochs=task.early_stop_epochs,
                    device=device,
                    subset_percentage=task.subset_percentage,
                    is_final=task.is_final,
                    epoch_callback=epoch_logger,
                )

                log_buffer.append(
                    f"[Worker-{worker_config.worker_id} / {device_name}] Individual {task.index} -> Accuracy: {accuracy:.4f}, Loss: {loss:.4f}"
                )

                result = Result(
                    index=task.index,
                    fitness=accuracy,
                    loss=loss,
                    status="SUCCESS",
                    log_lines=log_buffer,
                )

            except (CudaOutOfMemoryError, NumericalInstabilityError) as e:
                log_buffer.append(
                    f"[Worker-{worker_config.worker_id} / {device_name}] Controlled failure for individual {task.index}: {e}"
                )
                result = Result(
                    index=task.index,
                    fitness=0.0,
                    loss=float("inf"),
                    status="FAILURE",
                    error_message=str(e),
                    log_lines=log_buffer,
                )
            except Exception as e:
                log_buffer.append(
                    f"[Worker-{worker_config.worker_id} / {device_name}] Unexpected error for individual {task.index}: {e}"
                )
                result = Result(
                    index=task.index,
                    fitness=0.0,
                    loss=float("inf"),
                    status="FAILURE",
                    error_message=str(e),
                    log_lines=log_buffer,
                )

            worker_config.result_queue.put(result)

        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break
        except SystemExit:
            break

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    ):
        self.training_epochs = training_epochs
        self.early_stop_epochs = early_stop_epochs
        self.subset_percentage = subset_percentage
        self.execution_config = config["parallel_config"]["execution"]
        self.neural_network_config = config["neural_network_config"]

        self._workers: List[mp.Process] = []
        self._task_queue: mp.Queue | None = None
        self._result_queue: mp.Queue | None = None
        self._shutting_down = False
        self._cleaned_up = False

    def evaluate_population(
        self,
        population: List[Dict],
        stop_conditions: StopConditions,
        is_final: bool = False,
    ) -> Tuple[List[float], List[float]]:
        if self._shutting_down:
            logger.warning("Evaluator is shutting down, skipping evaluation")
            return [0.0] * len(population), [float("inf")] * len(population)

        pop_size = len(population)

        available_gpus = torch.cuda.device_count()
        gpu_workers_count = self.execution_config["gpu_workers"]
        cpu_workers_count = self.execution_config["cpu_workers"]

        if gpu_workers_count > available_gpus:
            logger.warning(
                f"Config requested {gpu_workers_count} GPU workers, but only {available_gpus} are available. Adjusting."
            )
            gpu_workers_count = available_gpus

        # Calculate optimal number of threads per process
        total_workers = gpu_workers_count + cpu_workers_count
        num_threads_per_process = (
            max(1, os.cpu_count() // total_workers) if total_workers > 0 else 1
        )

        ctx = mp.get_context("spawn")
        self._result_queue = ctx.Queue()
        self._task_queue = ctx.Queue()

        # Enqueue all tasks
        for i, ind in enumerate(population):
            self._task_queue.put(
                Task(
                    index=i,
                    neural_network_config=self.neural_network_config,
                    individual_hyperparams=ind,
                    training_epochs=self.training_epochs,
                    early_stop_epochs=self.early_stop_epochs,
                    subset_percentage=self.subset_percentage,
                    pop_size=pop_size,
                    is_final=is_final,
                )
            )

        # Add sentinel tasks to stop workers
        for _ in range(total_workers):
            self._task_queue.put(None)

        # Spawn workers
        workers = []
        for i in range(gpu_workers_count):
            w = WorkerConfig(
                i,
                i,
                self._task_queue,
                self._result_queue,
                num_threads_per_process,
            )
            p = ctx.Process(target=worker_main, args=(w,))
            p.start()
            workers.append(p)
        for i in range(cpu_workers_count):
            w = WorkerConfig(
                i + gpu_workers_count,
                "cpu",
                self._task_queue,
                self._result_queue,
                num_threads_per_process,
            )
            p = ctx.Process(target=worker_main, args=(w,))
            p.start()
            workers.append(p)

        self._workers = workers

        # Collect results
        results_list: List[Result] = [None] * pop_size
        finished = 0
        timeout_seconds = 300
        result_timeout_seconds = 600

        try:
            while finished < pop_size and not self._shutting_down:
                try:
                    # Check if any worker has died unexpectedly
                    for p in self._workers:
                        if not p.is_alive():
                            # Worker died without sending a result
                            raise RuntimeError
                        raise RuntimeError(
                            f"Worker process {p.pid} terminated unexpectedly. This may be due to an out-of-memory error or insufficient shared memory."
                        )

                    result: Result = self._result_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                results_list[result.index] = result
                finished += 1

                # Flush logs sequentially per individual
                for line in result.log_lines:
                    logger.info(line)

                should_stop, reason = stop_conditions.should_stop_evaluation(
                    result.fitness
                )

                if should_stop:
                    logger.info(reason)
                    break
        except RuntimeError as e:
            logger.error(
                f"A critical error occurred in the parallel evaluator: {e}"
            )
            self._shutting_down = True
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt detected. Cleaning up workers...")
            self._shutting_down = True
            self.cleanup_workers()
            raise
        finally:
            self.cleanup_workers()

        # If we're shutting down, don't return invalid results
        if self._shutting_down:
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
