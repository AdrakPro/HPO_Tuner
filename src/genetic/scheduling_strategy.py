"""
Defines the scheduling strategies for the ParallelEvaluator.

This module uses the Strategy design pattern to encapsulate different ways of
distributing the evaluation workload (e.g., CPU only, GPU only, Hybrid).
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List

import torch
import torch.multiprocessing as mp

from src.logger.logger import logger
from src.model.parallel import WorkerConfig
from src.parallel.worker import worker_main


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
    ) -> List[mp.Process]:
        """Helper to spawn and start worker processes."""
        workers = []
        total_workers = num_gpu_workers + num_cpu_workers
        if total_workers == 0:
            return []

        num_threads_per_process = max(1, os.cpu_count() // total_workers)

        # Spawn GPU workers
        for i in range(num_gpu_workers):
            w_config = WorkerConfig(
                worker_id=i,
                device=i,
                task_queue=task_queue,
                result_queue=result_queue,
                num_threads=num_threads_per_process,
                session_log_filename=session_log_filename,
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
                num_threads=num_threads_per_process,
                session_log_filename=session_log_filename,
            )
            p = ctx.Process(target=worker_main, args=(w_config,))
            p.start()
            workers.append(p)

        return workers


class CPUOnlyStrategy(SchedulingStrategy):
    """Schedules all tasks to be run on CPU workers."""

    def launch_workers(self, **kwargs) -> List[mp.Process]:
        logger.info("Using CPUOnly scheduling strategy.")
        num_cpu_workers = kwargs["execution_config"].get("cpu_workers", 1)

        return self._spawn_processes(
            ctx=kwargs["ctx"],
            task_queue=kwargs["task_queue"],
            result_queue=kwargs["result_queue"],
            num_cpu_workers=num_cpu_workers,
            session_log_filename=kwargs["session_log_filename"],
        )


class GPUOnlyStrategy(SchedulingStrategy):
    """Schedules all tasks to be run on GPU workers."""

    def launch_workers(self, **kwargs) -> List[mp.Process]:
        logger.info("Using GPUOnly scheduling strategy.")
        available_gpus = torch.cuda.device_count()
        num_gpu_workers = kwargs["execution_config"].get("gpu_workers", 0)

        if num_gpu_workers > available_gpus:
            logger.warning(
                f"Requested {num_gpu_workers} GPUs, but only {available_gpus} are available. Adjusting."
            )
            num_gpu_workers = available_gpus

        if num_gpu_workers == 0:
            logger.error(
                "GPUOnlyStrategy selected, but no GPU workers are configured or available."
            )
            return []

        return self._spawn_processes(
            ctx=kwargs["ctx"],
            task_queue=kwargs["task_queue"],
            result_queue=kwargs["result_queue"],
            num_gpu_workers=num_gpu_workers,
            session_log_filename=kwargs["session_log_filename"],
        )


class HybridStrategy(SchedulingStrategy):
    """Schedules tasks on both GPU and CPU workers."""

    def launch_workers(self, **kwargs) -> List[mp.Process]:
        logger.info("Using Hybrid scheduling strategy.")
        available_gpus = torch.cuda.device_count()
        num_gpu_workers = kwargs["execution_config"].get("gpu_workers", 0)
        num_cpu_workers = kwargs["execution_config"].get("cpu_workers", 0)

        if num_gpu_workers > available_gpus:
            logger.warning(
                f"Requested {num_gpu_workers} GPUs, but only {available_gpus} are available. Adjusting."
            )
            num_gpu_workers = available_gpus

        if num_gpu_workers + num_cpu_workers == 0:
            logger.error(
                "HybridStrategy selected, but no workers are configured."
            )
            return []

        return self._spawn_processes(
            ctx=kwargs["ctx"],
            task_queue=kwargs["task_queue"],
            result_queue=kwargs["result_queue"],
            num_gpu_workers=num_gpu_workers,
            num_cpu_workers=num_cpu_workers,
            session_log_filename=kwargs["session_log_filename"],
        )
