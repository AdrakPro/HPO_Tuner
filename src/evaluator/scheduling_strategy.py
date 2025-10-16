"""
Defines the scheduling strategies for the ParallelEvaluator.

This module uses the Strategy design pattern to encapsulate different ways of
distributing the evaluation workload (e.g., CPU only, GPU only, Hybrid).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

import torch
import torch.multiprocessing as mp

from src.evaluator.worker import worker_main
from src.logger.logger import logger
from src.model.parallel import WorkerConfig


class SchedulingStrategy(ABC):
    """
    Abstract base class for a scheduling strategy.
    Defines the interface for launching a persistent pool of worker processes.
    """

    @abstractmethod
    def launch_workers(
        self,
        ctx,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        execution_config: Dict,
        session_log_filename: str,
        fixed_batch_size: Optional[int] = None,
    ) -> List[mp.Process]:
        """
        The core method for a strategy. It must launch the necessary
        worker processes based on the execution config.

        Returns:
            A list of the started multiprocessing.Process objects.
        """
        pass

class CPUOnlyStrategy(SchedulingStrategy):
    """Schedules all tasks to be run on CPU workers."""

    def launch_workers(self, **kwargs) -> List[mp.Process]:
        logger.info("Using CPU-Only scheduling strategy.")
        exec_config = kwargs["execution_config"]
        num_cpu_workers = exec_config["cpu_workers"]

        dl_config = exec_config["dataloader_workers"]
        dl_per_cpu = dl_config["per_cpu"]

        return self._spawn_processes(
            ctx=kwargs["ctx"],
            task_queue=kwargs["task_queue"],
            result_queue=kwargs["result_queue"],
            num_cpu_workers=num_cpu_workers,
            session_log_filename=kwargs["session_log_filename"],
            dl_workers_per_cpu=dl_per_cpu,
            fixed_batch_size=kwargs["fixed_batch_size"],
        )


class GPUOnlyStrategy(SchedulingStrategy):
    """Schedules all tasks to be run on GPU workers."""

    def launch_workers(self, **kwargs) -> List[mp.Process]:
        logger.info("Using GPU-Only scheduling strategy.")

        available_gpus = torch.cuda.device_count()
        exec_config = kwargs["execution_config"]
        num_gpu_workers = kwargs["execution_config"]["gpu_workers"]

        if num_gpu_workers > available_gpus:
            logger.warning(
                f"Requested {num_gpu_workers} GPUs, but only {available_gpus} are available. Adjusting."
            )
            num_gpu_workers = available_gpus

        if num_gpu_workers == 0:
            logger.error(
                "GPU-Only Strategy selected, but no GPU workers are configured or available."
            )
            return []

        dl_config = exec_config["dataloader_workers"]
        dl_per_gpu = dl_config["per_gpu"]

        return self._spawn_processes(
            ctx=kwargs["ctx"],
            task_queue=kwargs["task_queue"],
            result_queue=kwargs["result_queue"],
            num_gpu_workers=num_gpu_workers,
            session_log_filename=kwargs["session_log_filename"],
            dl_workers_per_gpu=dl_per_gpu,
            fixed_batch_size=kwargs["fixed_batch_size"],
        )


class SimpleHybridStrategy(SchedulingStrategy):
    """Schedules tasks on both GPU and CPU workers."""

    def launch_workers(self, **kwargs) -> List[mp.Process]:
        logger.info("Using HYBRID scheduling strategy.")
        exec_config = kwargs["execution_config"]
        available_gpus = torch.cuda.device_count()
        num_gpu_workers = kwargs["execution_config"]["gpu_workers"]
        num_cpu_workers = kwargs["execution_config"]["cpu_workers"]

        if num_gpu_workers > available_gpus:
            logger.warning(
                f"Requested {num_gpu_workers} GPUs, but only {available_gpus} are available. Adjusting."
            )
            num_gpu_workers = available_gpus

        if num_gpu_workers + num_cpu_workers == 0:
            logger.error(
                "HYBRID Strategy selected, but no workers are configured."
            )
            return []

        dl_config = exec_config["dataloader_workers"]
        dl_per_gpu = dl_config["per_gpu"]
        dl_per_cpu = dl_config["per_cpu"]

        return self._spawn_processes(
            ctx=kwargs["ctx"],
            task_queue=kwargs["task_queue"],
            result_queue=kwargs["result_queue"],
            num_gpu_workers=num_gpu_workers,
            num_cpu_workers=num_cpu_workers,
            session_log_filename=kwargs["session_log_filename"],
            dl_workers_per_gpu=dl_per_gpu,
            dl_workers_per_cpu=dl_per_cpu,
        )

class GPUOnlyStrategy(SchedulingStrategy):
    """Schedules all tasks to be run on GPU workers."""

    def launch_workers(self, **kwargs: Any) -> Dict[str, Any]:
        logger.info("Using GPU-Only scheduling strategy.")
        ctx = kwargs["ctx"]
        task_queue = ctx.Queue()

        available_gpus = torch.cuda.device_count()
        exec_config = kwargs["execution_config"]
        num_gpu_workers = exec_config.get("gpu_workers", available_gpus)

        if num_gpu_workers > available_gpus:
            logger.warning(
                f"Requested {num_gpu_workers} GPUs, but only {available_gpus} are available. Adjusting."
            )
            num_gpu_workers = available_gpus

        if num_gpu_workers == 0:
            logger.error(
                "GPU-Only Strategy selected, but no GPU workers are configured or available."
            )
            return {"workers": [], "task_queues": []}

        dl_config = exec_config.get("dataloader_workers", {})
        dl_per_gpu = dl_config.get("per_gpu", 4)

        workers = _spawn_processes(
            ctx=ctx,
            task_queue=task_queue,
            result_queue=kwargs["result_queue"],
            num_gpu_workers=num_gpu_workers,
            session_log_filename=kwargs["session_log_filename"],
            dl_workers_per_gpu=dl_per_gpu,
        )
        return {"workers": workers, "task_queues": [task_queue]}


class HybridStrategy(SchedulingStrategy):
    """
    The main hybrid strategy. Implements an asymmetric scheduling mechanism 
    with task stealing to prevent CPU bottlenecking.
    - Creates separate queues for GPU (primary) and CPU (secondary) workers.
    - A 'task stealer' thread is expected to be managed by the Evaluator
      to move surplus tasks from the GPU queue to the CPU queue.
    """

    def launch_workers(self, **kwargs: Any) -> Dict[str, Any]:
        logger.info("Using main HYBRID strategy with Task Stealing.")
        ctx = kwargs["ctx"]
        gpu_task_queue = ctx.Queue(maxsize=1000)
        cpu_task_queue = ctx.Queue(maxsize=1000)

        exec_config = kwargs["execution_config"]
        available_gpus = torch.cuda.device_count()
        num_gpu_workers = exec_config.get("gpu_workers", available_gpus)
        num_cpu_workers = exec_config.get("cpu_workers", 0)

        if num_gpu_workers > available_gpus:
            logger.warning(
                f"Requested {num_gpu_workers} GPUs, but only {available_gpus} are available. Adjusting."
            )
            num_gpu_workers = available_gpus

        if num_gpu_workers + num_cpu_workers == 0:
            logger.error(
                "HybridStrategy selected, but no workers are configured."
            )
            return {"workers": [], "task_queues": []}

        dl_config = exec_config.get("dataloader_workers", {})
        dl_per_gpu = dl_config.get("per_gpu", 4)
        dl_per_cpu = dl_config.get("per_cpu", 1)
        
        gpu_workers = _spawn_processes(
            ctx=ctx,
            task_queue=gpu_task_queue,
            result_queue=kwargs["result_queue"],
            num_gpu_workers=num_gpu_workers,
            session_log_filename=kwargs["session_log_filename"],
            dl_workers_per_gpu=dl_per_gpu,
            fixed_batch_size=kwargs.get("fixed_batch_size")
        )

        cpu_workers = _spawn_processes(
            ctx=ctx,
            task_queue=cpu_task_queue,
            result_queue=kwargs["result_queue"],
            num_cpu_workers=num_cpu_workers,
            session_log_filename=kwargs["session_log_filename"],
            dl_workers_per_cpu=dl_per_cpu,
            gpu_worker_offset=num_gpu_workers, # To ensure unique worker_ids
            fixed_batch_size=kwargs.get("fixed_batch_size")
        )

        return {
            "workers": gpu_workers + cpu_workers,
            "gpu_task_queue": gpu_task_queue,
            "cpu_task_queue": cpu_task_queue,
        }

def _spawn_processes(
    ctx: Any,
    result_queue: mp.Queue,
    session_log_filename: str,
    task_queue: mp.Queue,
    num_gpu_workers: int = 0,
    num_cpu_workers: int = 0,
    dl_workers_per_gpu: int = 1,
    dl_workers_per_cpu: int = 1,
    fixed_batch_size: Optional[int] = None,
    gpu_worker_offset: int = 0,
) -> List[mp.Process]:
    """Helper to spawn and start worker processes."""
    workers = []

    # Spawn GPU workers
    for i in range(num_gpu_workers):
        w_config = WorkerConfig(
            worker_id=i + gpu_worker_offset,
            device=i,
            task_queue=task_queue,
            result_queue=result_queue,
            session_log_filename=session_log_filename,
            num_dataloader_workers=dl_workers_per_gpu,
            fixed_batch_size=fixed_batch_size,
        )
        p = ctx.Process(target=worker_main, args=(w_config,))
        p.start()
        workers.append(p)

    # Spawn CPU workers
    for i in range(num_cpu_workers):
        w_config = WorkerConfig(
            worker_id=i + num_gpu_workers + gpu_worker_offset,
            device="cpu",
            task_queue=task_queue,
            result_queue=result_queue,
            session_log_filename=session_log_filename,
            num_dataloader_workers=dl_workers_per_cpu,
            fixed_batch_size=fixed_batch_size,
            total_cpu_workers=num_cpu_workers,
        )
        p = ctx.Process(target=worker_main, args=(w_config,))
        p.start()
        workers.append(p)

    return workers
