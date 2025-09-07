"""
Contains the core logic for a single worker process.
This code is executed by each process in the multiprocessing pool.
"""

import os
import queue
import signal
import sys
import time
from typing import List, Union

import torch

from src.logger.logger import logger
from src.model.chromosome import Chromosome
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
            task: Task = worker_config.task_queue.get(timeout=1.0)
            if task is None:  # Sentinel to stop
                break

            log_buffer: List[str] = [
                f"[Worker-{worker_config.worker_id} / {device_name}] Evaluating Individual {task.index}/{task.pop_size} ({task.training_epochs} epochs)",
                f"[Worker-{worker_config.worker_id} / {device_name}] Hyperparameters: {task.individual_hyperparams}",
            ]

            try:
                start_time = time.time()
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

                finish_time = time.time()
                duration = finish_time - start_time

                log_buffer.append(
                    f"[Worker-{worker_config.worker_id} / {device_name}] Individual {task.index} -> Accuracy: {accuracy:.4f}, Loss: {loss:.4f} | Duration: {duration:.2f}s"
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
