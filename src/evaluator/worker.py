"""
Contains the core logic for a single worker process.
This code is executed by each process in the multiprocessing pool.
"""

import os
import queue
import signal
import sys
import time
from typing import List, Union, Tuple

import torch

from src.logger.logger import logger
from src.model.chromosome import Chromosome
from src.model.parallel import Result, Task, WorkerConfig
from src.nn.train_and_eval import train_and_eval
from src.utils.exceptions import CudaOutOfMemoryError, NumericalInstabilityError
from src.utils.thread_optimizer import (
    ThreadOptimizer,
)


def init_device(device_id: Union[str, int]) -> torch.device:
    """
    Initialize torch device for a worker with DGX A100 optimizations.
    """
    ThreadOptimizer.enable_tf32()

    if isinstance(device_id, int):
        # Set CUDA device after process spawn
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        if not torch.cuda.is_available():
            logger.error(f"CUDA not available for GPU worker {device_id}.")
            return torch.device("cpu")
        else:
            device_props = torch.cuda.get_device_properties(device_id)
            logger.info(
                f"GPU Worker {device_id}: {device_props.name}, "
                f"{device_props.total_memory / 1024**3:.1f} GB"
            )
            return torch.device("cuda")
    else:
        return torch.device("cpu")


def worker_main(worker_config: WorkerConfig) -> None:
    """
    Top-level worker function, so it can be pickled by multiprocessing.
    """
    if worker_config.device == "cpu":
        cores_per_worker = ThreadOptimizer.optimize_worker_threads(
            worker_id=worker_config.worker_id,
            total_workers=worker_config.total_cpu_workers,
            total_system_cores=128,
            reserved_cores=24,  # Reserve cores for system/GPU
        )

        os.environ["OMP_NUM_THREADS"] = str(cores_per_worker)
        os.environ["MKL_NUM_THREADS"] = str(cores_per_worker)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cores_per_worker)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cores_per_worker)

        torch.set_num_threads(cores_per_worker)
    else:
        # GPU workers need fewer CPU threads
        # TODO: 4 is recommended value, calculate and check if we have that many cores
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"
        torch.set_num_threads(4)

    thread_info = ThreadOptimizer.get_system_threading_info()

    log_buffer: List[str | Tuple[str, str]] = [
        f"Worker {worker_config.worker_id} thread config: {thread_info}"
    ]

    # Ignore SIGINT during imports to prevent KeyboardInterrupt during initialization
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

    device = init_device(worker_config.device)
    device_name = (
        f"GPU-{worker_config.device}"
        if isinstance(worker_config.device, int)
        else "CPU"
    )

    # Restore signal handler after imports are complete
    signal.signal(signal.SIGINT, original_sigint_handler)

    if device.type == "cuda":
        log_buffer.append(
            (
                f"Worker {worker_config.worker_id} using {torch.cuda.get_device_name(device)} with {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f}GB memory",
                "file_only",
            )
        )

    while True:
        try:
            task = worker_config.task_queue.get(timeout=1.0)

            if task is None:  # Sentinel to stop
                break

            log_buffer.append(
                f"[Worker-{worker_config.worker_id} / {device_name}] Evaluating Individual {task.index}/{task.pop_size} ({task.training_epochs} epochs)"
            )

            log_buffer.append(
                (
                    f"[Worker-{worker_config.worker_id} / {device_name}] Hyperparameters: {task.individual_hyperparams}",
                    "file_only",
                )
            )

            try:
                start_time = time.perf_counter()
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
                    train_indices=task.train_indices,
                    test_indices=task.test_indices,
                    num_dataloader_workers=worker_config.num_dataloader_workers,
                )

                duration = time.perf_counter() - start_time

                log_buffer.append(
                    f"[Worker-{worker_config.worker_id} / {device_name}] Individual {task.index} -> "
                    f"Accuracy: {accuracy:.4f}, Loss: {loss:.4f} | Duration: {duration:.2f}s"
                )

                if device.type == "cuda":
                    gpu_usage = (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                    )
                    log_buffer.append(
                        f"[Worker-{worker_config.worker_id} / {device_name}] GPU memory used: {gpu_usage:.2f} GB"
                    )
                    torch.cuda.reset_peak_memory_stats(device)

                result = Result(
                    index=task.index,
                    fitness=accuracy,
                    loss=loss,
                    duration_seconds=duration,
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
                    duration_seconds=0.0,
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
                    duration_seconds=0.0,
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

    # Reset signal handler
    signal.signal(signal.SIGINT, original_sigint_handler)
