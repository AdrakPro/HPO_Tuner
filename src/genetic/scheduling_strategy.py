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
from src.model.parallel import Task, WorkerConfig
from src.parallel.worker import worker_main


class SchedulingStrategy(ABC):
    """
    Abstract base class for a scheduling strategy.
    Defines the interface for distributing tasks and launching workers.
    """

    @abstractmethod
    def distribute_and_run(
        self,
        population: List[Dict],
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        neural_network_config: Dict,
        execution_config: Dict,
        training_epochs: int,
        early_stop_epochs: int,
        subset_percentage: float,
        is_final: bool = False,
    ) -> List[mp.Process]:
        """
        The core method for a strategy. It must populate the task queue
        and launch the necessary worker processes.

        Returns:
            A list of the started multiprocessing.Process objects.
        """
        pass

    @staticmethod
    def _enqueue_tasks(
        population: List[Dict],
        task_queue: mp.Queue,
        neural_network_config: Dict,
        training_epochs: int,
        early_stop_epochs: int,
        subset_percentage: float,
        is_final: bool,
    ) -> None:
        """Helper to populate the task queue."""
        pop_size = len(population)
        for i, ind in enumerate(population):
            task_queue.put(
                Task(
                    index=i,
                    neural_network_config=neural_network_config,
                    individual_hyperparams=ind,
                    training_epochs=training_epochs,
                    early_stop_epochs=early_stop_epochs,
                    subset_percentage=subset_percentage,
                    pop_size=pop_size,
                    is_final=is_final,
                )
            )

    @staticmethod
    def _launch_workers(
        ctx,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        num_gpu_workers: int = 0,
        num_cpu_workers: int = 0,
    ) -> List[mp.Process]:
        """Helper to spawn worker processes."""
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
            )
            p = ctx.Process(target=worker_main, args=(w_config,))
            p.start()
            workers.append(p)

        return workers


class CPUOnlyStrategy(SchedulingStrategy):
    """Schedules all tasks to be run on CPU workers."""

    def distribute_and_run(self, **kwargs) -> List[mp.Process]:
        ctx = mp.get_context("spawn")
        task_queue = kwargs["task_queue"]

        self._enqueue_tasks(
            population=kwargs["population"],
            task_queue=task_queue,
            neural_network_config=kwargs["neural_network_config"],
            training_epochs=kwargs["training_epochs"],
            early_stop_epochs=kwargs["early_stop_epochs"],
            subset_percentage=kwargs["subset_percentage"],
            is_final=kwargs["is_final"],
        )

        num_cpu_workers = kwargs["execution_config"].get("cpu_workers", 1)

        # Add sentinel tasks
        for _ in range(num_cpu_workers):
            task_queue.put(None)

        return self._launch_workers(
            ctx=ctx,
            task_queue=task_queue,
            result_queue=kwargs["result_queue"],
            num_cpu_workers=num_cpu_workers,
        )


class GPUOnlyStrategy(SchedulingStrategy):
    """Schedules all tasks to be run on GPU workers."""

    def distribute_and_run(self, **kwargs) -> List[mp.Process]:
        ctx = mp.get_context("spawn")
        task_queue = kwargs["task_queue"]

        self._enqueue_tasks(
            population=kwargs["population"],
            task_queue=task_queue,
            neural_network_config=kwargs["neural_network_config"],
            training_epochs=kwargs["training_epochs"],
            early_stop_epochs=kwargs["early_stop_epochs"],
            subset_percentage=kwargs["subset_percentage"],
            is_final=kwargs["is_final"],
        )

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

        # Add sentinel tasks
        for _ in range(num_gpu_workers):
            task_queue.put(None)

        return self._launch_workers(
            ctx=ctx,
            task_queue=task_queue,
            result_queue=kwargs["result_queue"],
            num_gpu_workers=num_gpu_workers,
        )


class HybridStrategy(SchedulingStrategy):
    """Schedules tasks on both GPU and CPU workers."""

    def distribute_and_run(self, **kwargs) -> List[mp.Process]:
        ctx = mp.get_context("spawn")
        task_queue = kwargs["task_queue"]

        self._enqueue_tasks(
            population=kwargs["population"],
            task_queue=task_queue,
            neural_network_config=kwargs["neural_network_config"],
            training_epochs=kwargs["training_epochs"],
            early_stop_epochs=kwargs["early_stop_epochs"],
            subset_percentage=kwargs["subset_percentage"],
            is_final=kwargs["is_final"],
        )

        available_gpus = torch.cuda.device_count()
        num_gpu_workers = kwargs["execution_config"].get("gpu_workers", 0)
        num_cpu_workers = kwargs["execution_config"].get("cpu_workers", 0)

        if num_gpu_workers > available_gpus:
            logger.warning(
                f"Requested {num_gpu_workers} GPUs, but only {available_gpus} are available. Adjusting."
            )
            num_gpu_workers = available_gpus

        total_workers = num_gpu_workers + num_cpu_workers

        if total_workers == 0:
            logger.error(
                "HybridStrategy selected, but no workers are configured."
            )
            return []

        # Add sentinel tasks
        for _ in range(total_workers):
            task_queue.put(None)

        return self._launch_workers(
            ctx=ctx,
            task_queue=task_queue,
            result_queue=kwargs["result_queue"],
            num_gpu_workers=num_gpu_workers,
            num_cpu_workers=num_cpu_workers,
        )
