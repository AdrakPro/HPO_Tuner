import os
import queue
import signal
import time
import traceback
from typing import Union, List

from torch import cuda
from torch import device, set_num_threads, set_num_interop_threads

from src.model.chromosome import Chromosome, OptimizerSchedule
from src.model.parallel import Result, WorkerConfig
from src.nn.data_loader import get_dataset_loaders
from src.nn.train_and_eval import train_and_eval
from src.utils.exceptions import NumericalInstabilityError
from src.utils.thread_optimizer import ThreadOptimizer


def pin_worker_to_cores(core_ids: List[int]):
    """Pins the current process to the specified core IDs."""
    if not core_ids:
        return
    try:
        os.sched_setaffinity(0, core_ids)
    except Exception:
        pass


def init_device(device_id: Union[str, int]) -> device:
    """
    Initialize torch device for a worker.
    """
    ThreadOptimizer.enable_tf32()

    if isinstance(device_id, int):
        if not cuda.is_available():
            return device("cpu")
        else:
            cuda.set_device(device_id)
            return device(f"cuda:{device_id}")
    else:
        return device("cpu")


def worker_main(worker_config: WorkerConfig) -> None:
    """
    Top-level worker function, so it can be pickled by multiprocessing.
    """
    # Ignore SIGINT during imports to prevent KeyboardInterrupt during initialization
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

    assigned_cores = getattr(worker_config, "core_ids", [])
    worker_type = getattr(worker_config, "type", "unknown")
    device_spec = getattr(worker_config, "device", "cpu")
    device_info = (
        f"CPU" if worker_type == "cpu" else f"GPU device={device_spec}"
    )

    if assigned_cores:
        pin_worker_to_cores(assigned_cores)

    # TODO make settings threads per worker in config cpu/gpu
    if worker_type == "gpu":
        num_compute_threads = 1
    elif worker_type == "cpu":
        num_compute_threads = 14
    else:
        num_compute_threads = 1

    set_num_interop_threads(1)
    set_num_threads(num_compute_threads)

    dev = init_device(worker_config.device)
    device_name = (
        f"GPU-{worker_config.device}"
        if isinstance(worker_config.device, int)
        else "CPU"
    )

    # Restore signal handler after imports are complete
    signal.signal(signal.SIGINT, original_sigint_handler)

    while True:
        try:
            task = worker_config.task_queue.get(timeout=1.0)

            if task is None:  # Sentinel to stop
                break

            log_buffer = [
                f"[Worker-{worker_config.worker_id} / {device_name}] Evaluating Individual {task.index}/{task.pop_size} ({task.training_epochs} epochs)",
                (
                    f"[Worker-{worker_config.worker_id} / {device_name}] Hyperparameters: {task.individual_hyperparams}",
                    "file_only",
                ),
            ]
            try:
                start_time = time.perf_counter()
                chromosome = Chromosome.from_dict(task.individual_hyperparams)

                # Scaling LR based on optimizer
                # TODO set REFERENCE_BATCH in config
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

                    if scaled_lr < original_lr * scale_factor:
                        log_buffer.append(
                            f"[Worker-{worker_config.worker_id} / {device_name}] "
                            f"Scaled base_lr from {original_lr:.6f} to {original_lr * scale_factor:.6f}, "
                            f"but CAPPED at {scaled_lr:.6f} for batch_size {chromosome.batch_size}"
                        )
                    else:
                        log_buffer.append(
                            f"[Worker-{worker_config.worker_id} / {device_name}] "
                            f"Scaled base_lr to {chromosome.base_lr:.6f} for batch_size {chromosome.batch_size}"
                        )
                else:
                    log_buffer.append(
                        f"[Worker-{worker_config.worker_id} / {device_name}] "
                        f"Using base_lr {chromosome.base_lr:.6f} for batch_size {chromosome.batch_size}"
                    )

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

                with get_dataset_loaders(
                    batch_size=chromosome.batch_size,
                    aug_intensity=chromosome.aug_intensity,
                    is_gpu=dev.type == "cuda",
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
                        device=dev,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        is_final=task.is_final,
                        epoch_callback=epoch_logger,
                    )

                duration = time.perf_counter() - start_time

                log_buffer.append(
                    f"[Worker-{worker_config.worker_id} / {device_name}] Individual {task.index} -> "
                    f"Accuracy: {accuracy:.4f}, Loss: {loss:.4f} | Duration: {duration:.2f}s"
                )

                if dev.type == "cuda":
                    gpu_usage = cuda.max_memory_allocated(dev) / 1024**3
                    log_buffer.append(
                        f"[Worker-{worker_config.worker_id} / {device_name}] GPU memory used: {gpu_usage:.2f} GB"
                    )
                    cuda.reset_peak_memory_stats(dev)

                result = Result(
                    index=task.index,
                    fitness=accuracy,
                    loss=loss,
                    duration_seconds=duration,
                    status="SUCCESS",
                    log_lines=log_buffer,
                    worker_type=worker_config.type,
                )

            except NumericalInstabilityError as e:
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
                    worker_type=worker_config.type,
                )
            except Exception as e:
                tb_str = traceback.format_exc()
                log_buffer.append(
                    f"[Worker-{worker_config.worker_id} / {device_name}] Unexpected error for individual {task.index}: {e} -> {tb_str}"
                )
                result = Result(
                    index=task.index,
                    fitness=0.0,
                    loss=float("inf"),
                    duration_seconds=0.0,
                    status="FAILURE",
                    error_message=str(e),
                    log_lines=log_buffer,
                    worker_type=worker_config.type,
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
