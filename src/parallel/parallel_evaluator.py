"""
Handles the parallel evaluation of a population of individuals.
"""

import os
import queue
import signal
import sys
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
            # logger.info(f"Worker will use GPU {device_id}.")
            return torch.device("cuda")
    else:
        # logger.info("Worker will use CPU.")
        return torch.device("cpu")


def worker_main(worker_config: WorkerConfig) -> None:
    """
    Top-level worker function so it can be pickled by multiprocessing.
    """
    def sigint_handler(signum, frame):
        sys.exit(1)  # Graceful exit on SIGINT

    signal.signal(signal.SIGINT, sigint_handler)

    # Set number of threads to avoid CPU oversubscription
    torch.set_num_threads(worker_config.num_threads)
    device = init_device(worker_config.device)
    device_name = (
        f"GPU-{worker_config.device}"
        if isinstance(worker_config.device, int)
        else "CPU"
    )

    while True:
        task: Task = worker_config.task_queue.get()
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

    def evaluate_population(
        self,
        population: List[Dict],
        stop_conditions: StopConditions,
        is_final: bool = False,
    ) -> Tuple[List[float], List[float]]:
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

        try:
            while finished < pop_size:
                try:
                    result: Result = self._result_queue.get(
                        timeout=timeout_seconds
                    )
                except queue.Empty:
                    logger.error(
                        "Timeout waiting for result from worker. Some tasks may have failed or hung."
                    )
                    break

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
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt detected. Cleaning up workers...")
            self.cleanup_workers()
            raise
        finally:
            self.cleanup_workers()

        fitness_scores = [r.fitness if r else 0.0 for r in results_list]
        loss_scores = [r.loss if r else float("inf") for r in results_list]
        return fitness_scores, loss_scores

    def set_training_epochs(self, epochs: int) -> None:
        self.training_epochs = epochs

    def set_subset_percentage(self, subset_percentage: float) -> None:
        """Updates the subset percentage."""
        self.subset_percentage = subset_percentage

    def cleanup_workers(self):
        logger.info("Cleaning up ParallelEvaluator workers...")
        if self._workers:
            for p in self._workers:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
            self._workers = []
