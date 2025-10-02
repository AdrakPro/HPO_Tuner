"""
Defines the scheduling strategies for the ParallelEvaluator.

This module uses the Strategy design pattern to encapsulate different ways of
distributing the evaluation workload (e.g., CPU only, GPU only, Hybrid).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

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

    @staticmethod
    def _spawn_processes(
        ctx,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        session_log_filename: str,
        num_gpu_workers: int = 0,
        num_cpu_workers: int = 0,
        dl_workers_per_gpu: int = 1,
        dl_workers_per_cpu: int = 1,
        fixed_batch_size: Optional[int] = None,
    ) -> List[mp.Process]:
        """Helper to spawn and start worker processes."""
        workers = []
        total_workers = num_gpu_workers + num_cpu_workers

        if total_workers == 0:
            return []

        # Spawn GPU workers
        for i in range(num_gpu_workers):
            w_config = WorkerConfig(
                worker_id=i,
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
                worker_id=i + num_gpu_workers,
                device="cpu",
                task_queue=task_queue,
                result_queue=result_queue,
                session_log_filename=session_log_filename,
                num_dataloader_workers=dl_workers_per_cpu,
                fixed_batch_size=fixed_batch_size,
            )
            p = ctx.Process(target=worker_main, args=(w_config,))
            p.start()
            workers.append(p)

        return workers


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


class HybridStrategy(SchedulingStrategy):
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
