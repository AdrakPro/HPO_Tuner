"""
Contains the core logic for a single worker process.
This code is executed by each process in the multiprocessing pool.
"""

import os
import queue
import signal
import time
from typing import Union

# --- CHANGE START: Import the logging module ---
# We need this module to create a logger instance inside the worker process.
import logging
# --- CHANGE END ---

import torch

from src.model.chromosome import Chromosome, OptimizerSchedule
from src.model.parallel import Result, WorkerConfig
from src.nn.data_loader import get_dataset_loaders
from src.nn.train_and_eval import train_and_eval
from src.utils.exceptions import CudaOutOfMemoryError, NumericalInstabilityError
from src.utils.thread_optimizer import (
    ThreadOptimizer,
)


def init_device(device_id: Union[str, int]) -> torch.device:
    """
    Initialize torch device for a worker.
    """
    ThreadOptimizer.enable_tf32()

    if isinstance(device_id, int):
        if not torch.cuda.is_available():
            # This will now be written to the log file by the worker's logger
            return torch.device("cpu")
        else:
            torch.cuda.set_device(device_id)
            return torch.device(f"cuda:{device_id}")
    else:
        return torch.device("cpu")


def worker_main(worker_config: WorkerConfig) -> None:
    """
    Top-level worker function, so it can be pickled by multiprocessing.
    """
    os.environ["OMP_SCHEDULE"] = "STATIC"
    os.environ["OMP_PROC_BIND"] = "CLOSE"

    if worker_config.device == "cpu":
        num_threads = 6
        # Pin the CPU worker to 6 cores (e.g., 0 through 5)
        cores_to_use = set(range(num_threads))
    else: # For GPU worker
        num_threads = 2
        # Pin the GPU worker to 2 cores (e.g., 0 and 1)
        cores_to_use = set(range(num_threads))

    try:
        pid = os.getpid()
        os.sched_setaffinity(pid, cores_to_use)
        print(f"Worker process {pid} (Device: {worker_config.device}, ID: {worker_config.worker_id}) pinned to CPUs: {os.sched_getaffinity(pid)}")
    except (AttributeError, FileNotFoundError):
        print("CPU affinity not supported on this system.")
    except Exception as e:
        print(f"Warning: Could not set worker affinity for PID {pid}: {e}")


    # Ignore SIGINT during imports to prevent KeyboardInterrupt during initialization
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

    device = init_device(worker_config.device)
    device_name = (
        f"GPU-{worker_config.device}"
        if isinstance(worker_config.device, int)
        else "CPU"
    )

    # --- CHANGE START: Set up a dedicated logger for this worker ---
    # This replaces the queue-based logging system.
    # It creates a new logger that writes directly to a dynamically constructed log file path.
    # This is ONLY safe for a single worker process.
    worker_logger = logging.getLogger(f"worker_{worker_config.worker_id}")
    worker_logger.setLevel(logging.INFO)
    worker_logger.propagate = False
    
    if not worker_logger.hasHandlers():
        # Construct the log file path as requested.
        log_dir = "logs"
        username = "AdrakPro"
        timestamp = "2025-10-21_09-35-44" # Formatted from '2025-10-21 09:35:44'
        
        # Ensure the directory exists.
        os.makedirs(log_dir, exist_ok=True)
        
        log_filename = os.path.join(log_dir, f"{username}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_filename)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        worker_logger.addHandler(file_handler)
    # --- CHANGE END ---

    # --- CHANGE START: Check for CUDA availability using the new logger ---
    if isinstance(worker_config.device, int) and not torch.cuda.is_available():
        worker_logger.error(f"CUDA not available for GPU worker {worker_config.device}.")
    # --- CHANGE END ---

    # Restore signal handler after imports are complete
    signal.signal(signal.SIGINT, original_sigint_handler)

    while True:
        try:
            task = worker_config.task_queue.get(timeout=1.0)

            if task is None:  # Sentinel to stop
                break

            # This buffer is for the final summary attached to the Result object
            summary_log_buffer = []

            # --- CHANGE START: Use the worker_logger directly ---
            # Replaces the old `log_message` function that used a queue.
            worker_logger.info(
                f"[Worker-{worker_config.worker_id} / {device_name}] Evaluating Individual {task.index}/{task.pop_size} ({task.training_epochs} epochs)"
            )
            # For file-only logs, we just log it. The main logger isn't involved.
            worker_logger.info(
                f"[Worker-{worker_config.worker_id} / {device_name}] Hyperparameters: {task.individual_hyperparams}"
            )
            # --- CHANGE END ---

            try:
                start_time = time.perf_counter()
                chromosome = Chromosome.from_dict(task.individual_hyperparams)

                # Scaling LR based on optimizer
                REFERENCE_BATCH = 128
                original_lr = chromosome.base_lr
                scale_factor = 0.0

                if REFERENCE_BATCH != chromosome.batch_size:
                    if _is_adamw(chromosome.optimizer_schedule):
                        scale_factor = (
                            chromosome.batch_size / REFERENCE_BATCH
                        ) ** 0.5
                        scaled_lr = chromosome.base_lr * scale_factor
                        max_lr = 0.01
                        scaled_lr = min(scaled_lr, max_lr)

                    elif _is_sgd(chromosome.optimizer_schedule):
                        scale_factor = chromosome.batch_size / REFERENCE_BATCH
                        scaled_lr = chromosome.base_lr * scale_factor
                        max_lr = 0.5
                        scaled_lr = min(scaled_lr, max_lr)
                    else:
                        scaled_lr = chromosome.base_lr

                    chromosome.base_lr = scaled_lr

                    # --- CHANGE START: Use worker_logger for all log messages ---
                    if scaled_lr < original_lr * scale_factor:
                        worker_logger.info(
                            f"[Worker-{worker_config.worker_id} / {device_name}] "
                            f"Scaled base_lr from {original_lr:.6f} to {original_lr * scale_factor:.6f}, "
                            f"but CAPPED at {scaled_lr:.6f} for batch_size {chromosome.batch_size}"
                        )
                    else:
                        worker_logger.info(
                            f"[Worker-{worker_config.worker_id} / {device_name}] "
                            f"Scaled base_lr to {chromosome.base_lr:.6f} for batch_size {chromosome.batch_size}"
                        )
                else:
                    worker_logger.info(
                        f"[Worker-{worker_config.worker_id} / {device_name}] "
                        f"Using base_lr {chromosome.base_lr:.6f} for batch_size {chromosome.batch_size}"
                    )
                    # --- CHANGE END ---

                def epoch_logger(
                    epoch,
                    train_acc,
                    train_loss,
                    test_acc,
                    test_loss,
                    early_stop=False,
                    final_msg=None,
                ):
                    # --- CHANGE START: The epoch callback now uses the worker_logger ---
                    if final_msg:
                        worker_logger.info(
                            f"[Worker-{worker_config.worker_id} / {device_name}] {final_msg}"
                        )
                        return

                    line = f"[Worker-{worker_config.worker_id} / {device_name}] Epoch {epoch}/{task.training_epochs} | Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"

                    if early_stop:
                        line += " / Early stopping triggered"
                    worker_logger.info(line)
                    # --- CHANGE END ---

                with get_dataset_loaders(
                    batch_size=chromosome.batch_size,
                    aug_intensity=chromosome.aug_intensity,
                    is_gpu=device.type == "cuda",
                    num_dataloader_workers=worker_config.num_dataloader_workers,
                    subset_percentage=task.subset_percentage,
                    train_indices=task.train_indices,
                    test_indices=task.test_indices,
                ) as (train_loader, test_loader):
                    accuracy, loss = train_and_eval(
                        chromosome=chromosome,
                        neural_config=task.neural_network_config,
                        epochs=task.training_epochs,
                        early_stop_epochs=task.early_stop_epochs,
                        device=device,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        is_final=task.is_final,
                        epoch_callback=epoch_logger,
                    )

                duration = time.perf_counter() - start_time

                summary_msg = (f"[Worker-{worker_config.worker_id} / {device_name}] Individual {task.index} -> "
                               f"Accuracy: {accuracy:.4f}, Loss: {loss:.4f} | Duration: {duration:.2f}s")
                # --- CHANGE START: Log the summary message and also add to buffer ---
                worker_logger.info(summary_msg)
                summary_log_buffer.append(summary_msg)
                # --- CHANGE END ---

                if device.type == "cuda":
                    gpu_usage = (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                    )
                    gpu_msg = f"[Worker-{worker_config.worker_id} / {device_name}] GPU memory used: {gpu_usage:.2f} GB"
                    # --- CHANGE START: Log GPU message and also add to buffer ---
                    worker_logger.info(gpu_msg)
                    summary_log_buffer.append(gpu_msg)
                    # --- CHANGE END ---
                    torch.cuda.reset_peak_memory_stats(device)

                result = Result(
                    index=task.index,
                    fitness=accuracy,
                    loss=loss,
                    duration_seconds=duration,
                    status="SUCCESS",
                    # The Result object still contains the final summary lines
                    log_lines=summary_log_buffer,
                    worker_type=worker_config.type
                )

            except (CudaOutOfMemoryError, NumericalInstabilityError) as e:
                error_msg = f"[Worker-{worker_config.worker_id} / {device_name}] Controlled failure for individual {task.index}: {e}"
                # --- CHANGE START: Log errors directly ---
                worker_logger.error(error_msg)
                # --- CHANGE END ---
                result = Result(
                    index=task.index,
                    fitness=0.0,
                    loss=float("inf"),
                    duration_seconds=0.0,
                    status="FAILURE",
                    error_message=str(e),
                    log_lines=[error_msg],
                    worker_type=worker_config.type
                )
            except Exception as e:
                error_msg = f"[Worker-{worker_config.worker_id} / {device_name}] Unexpected error for individual {task.index}: {e}"
                # --- CHANGE START: Log errors directly ---
                worker_logger.error(error_msg, exc_info=True) # exc_info adds traceback
                # --- CHANGE END ---
                result = Result(
                    index=task.index,
                    fitness=0.0,
                    loss=float("inf"),
                    duration_seconds=0.0,
                    status="FAILURE",
                    error_message=str(e),
                    log_lines=[error_msg],
                    worker_type=worker_config.type
                )

            worker_config.result_queue.put(result)

        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break
        except SystemExit:
            break

    # --- CHANGE START: Clean up the logger handler ---
    # This is good practice to release the file handle.
    if worker_logger and worker_logger.hasHandlers():
        for handler in worker_logger.handlers[:]:
            handler.close()
            worker_logger.removeHandler(handler)
    # --- CHANGE END ---

    # Reset signal handler
    signal.signal(signal.SIGINT, original_sigint_handler)


def _is_adamw(optimizer_schedule: OptimizerSchedule):
    return optimizer_schedule in [
        OptimizerSchedule.ADAMW_COSINE,
        OptimizerSchedule.ADAMW_ONECYCLE,
        OptimizerSchedule.ADAMW_EXPONENTIAL,
    ]


def _is_sgd(optimizer_schedule: OptimizerSchedule):
    return optimizer_schedule in [
        OptimizerSchedule.SGD_COSINE,
        OptimizerSchedule.SGD_ONECYCLE,
        OptimizerSchedule.SGD_EXPONENTIAL,
    ]
