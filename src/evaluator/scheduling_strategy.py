"""
Defines the scheduling strategies for the ParallelEvaluator.

This module uses the Strategy design pattern to encapsulate different ways of
distributing the evaluation workload (e.g., CPU only, GPU only, Hybrid).
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

import torch
import torch.multiprocessing as mp

from src.evaluator.worker import worker_main
from src.logger.logger import logger
from src.model.parallel import WorkerConfig

# TODO make this configurable
# --- Configuration Constants ---
TOTAL_CORES = 64
CORES_PER_NODE = 16
GPU_TO_NODE_MAP = {0: 1, 1: 0, 2: 3, 3: 2}


class SchedulingStrategy(ABC):
    """
    Abstract base class for a scheduling strategy.
    Defines the interface for launching a persistent pool of worker processes.
    """

    @abstractmethod
    def launch_workers(
        self,
        ctx,
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


class CPUOnlyStrategy(SchedulingStrategy):
    """Schedules all tasks to be run on CPU workers."""

    # TODO: check for cpu count if we dont assign more than it is (what about hyperthreading?)
    def launch_workers(self, **kwargs) -> List[mp.Process]:
        logger.info("Using CPU-Only scheduling strategy.")
        ctx = kwargs["ctx"]
        task_queue = ctx.Queue()
        exec_config = kwargs["execution_config"]
        num_cpu_workers = exec_config["cpu_workers"]

        dl_config = exec_config["dataloader_workers"]
        dl_per_cpu = dl_config["per_cpu"]

        workers = _spawn_processes(
            ctx=kwargs["ctx"],
            task_queue=kwargs["task_queue"],
            result_queue=kwargs["result_queue"],
            num_cpu_workers=num_cpu_workers,
            session_log_filename=kwargs["session_log_filename"],
            dl_workers_per_cpu=dl_per_cpu,
        )

        return {"workers": workers, "task_queues": [task_queue]}


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
        )

        cpu_workers = _spawn_processes(
            ctx=ctx,
            task_queue=cpu_task_queue,
            result_queue=kwargs["result_queue"],
            num_cpu_workers=num_cpu_workers,
            session_log_filename=kwargs["session_log_filename"],
            dl_workers_per_cpu=dl_per_cpu,
            gpu_worker_offset=num_gpu_workers,
        )

        return {
            "workers": gpu_workers + cpu_workers,
            "gpu_task_queue": gpu_task_queue,
            "cpu_task_queue": cpu_task_queue,
        }


class NumaCoreAllocator:
    """
    Manages core availability and allocation to prevent conflicts.
    Ensures atomic allocation (all requested cores or none).
    """

    def __init__(
        self,
        total_cores: int = TOTAL_CORES,
        cores_per_node: int = CORES_PER_NODE,
    ):
        self.total_cores = total_cores
        self.assigned_mask = [False] * total_cores
        # Pre-calculate node ranges: {0: [0..15], 1: [16..31], ...}
        self.node_cores = {
            i: list(range(i * cores_per_node, (i + 1) * cores_per_node))
            for i in range(total_cores // cores_per_node)
        }

    def allocate(
        self, node_id: int, count: int, from_end: bool = False
    ) -> Optional[List[int]]:
        """
        Attempts to allocate 'count' cores from 'node_id'.
        Returns list of core_ids on success, or None on failure.
        """
        if node_id not in self.node_cores:
            logger.error(f"Invalid NUMA Node ID: {node_id}")
            return None

        available_cores = self.node_cores[node_id]

        if len(available_cores) < count:
            logger.error(
                f"Node {node_id} has insufficient total cores for request of {count}."
            )
            return None

        # Select candidates based on strategy (First available vs Last available)
        candidates = (
            available_cores[-count:] if from_end else available_cores[:count]
        )

        if any(self.assigned_mask[c] for c in candidates):
            logger.error(
                f"Core conflict detected on Node {node_id} for requested cores {candidates}."
            )
            return None

        for c in candidates:
            self.assigned_mask[c] = True

        return candidates


def _spawn_single_worker(ctx: Any, config: WorkerConfig) -> mp.Process:
    """Boilerplate wrapper to create and start a process."""
    p = ctx.Process(target=worker_main, args=(config,))
    p.start()
    return p


def _spawn_processes(
    ctx: Any,
    result_queue: mp.Queue,
    session_log_filename: str,
    task_queue: mp.Queue,
    num_gpu_workers: int = 0,
    num_cpu_workers: int = 0,
    dl_workers_per_gpu: int = 1,
    dl_workers_per_cpu: int = 1,
    gpu_worker_offset: int = 0,
) -> List[mp.Process]:
    """
    Spawns GPU and CPU workers with NUMA-aware core pinning.
    """
    workers = []
    allocator = NumaCoreAllocator()

    # TODO make this configurable
    CORES_FOR_CPU_WORKER = 14
    CORES_FOR_GPU_WORKER = 2

    if num_gpu_workers <= 0 and num_cpu_workers <= 0:
        return workers

    # --- 1. Spawn GPU Workers ---
    if num_gpu_workers > 0:
        logger.info(
            f"Spawning {num_gpu_workers} GPU workers ({CORES_FOR_GPU_WORKER} cores each)..."
        )

        for i in range(num_gpu_workers):
            device_id = i

            if device_id not in GPU_TO_NODE_MAP:
                logger.error(f"No NUMA mapping for GPU {device_id}. Skipping.")
                continue

            node_id = GPU_TO_NODE_MAP[device_id]

            core_ids = allocator.allocate(
                node_id, CORES_FOR_GPU_WORKER, from_end=True
            )

            if not core_ids:
                logger.warning(
                    f"Failed to allocate cores for GPU {device_id} on Node {node_id}."
                )
                continue

            worker_id = device_id + gpu_worker_offset
            logger.info(
                f"  Worker-{worker_id} (GPU-{device_id}) -> Node {node_id} (Cores {core_ids})"
            )

            w_config = WorkerConfig(
                worker_id=worker_id,
                device=device_id,
                core_ids=core_ids,
                task_queue=task_queue,
                result_queue=result_queue,
                session_log_filename=session_log_filename,
                num_dataloader_workers=dl_workers_per_gpu,
            )
            workers.append(_spawn_single_worker(ctx, w_config))

        time.sleep(1.0)

    # --- 2. Spawn CPU Workers ---
    if num_cpu_workers > 0:
        logger.info(
            f"Spawning {num_cpu_workers} CPU workers ({CORES_FOR_CPU_WORKER} cores each)..."
        )

        node_ids = sorted(allocator.node_cores.keys())

        for i in range(num_cpu_workers):
            node_id = node_ids[i % len(node_ids)]

            core_ids = allocator.allocate(
                node_id, CORES_FOR_CPU_WORKER, from_end=False
            )

            if not core_ids:
                logger.warning(
                    f"Failed to allocate cores for CPU Worker {i} on Node {node_id}."
                )
                continue

            worker_id = i
            logger.info(
                f"  Worker-{worker_id} (CPU) -> Node {node_id} (Cores {core_ids})"
            )

            w_config = WorkerConfig(
                worker_id=worker_id,
                device="cpu",
                core_ids=core_ids,
                task_queue=task_queue,
                result_queue=result_queue,
                session_log_filename=session_log_filename,
                num_dataloader_workers=dl_workers_per_cpu,
                total_cpu_workers=num_cpu_workers,
            )
            workers.append(_spawn_single_worker(ctx, w_config))

    logger.info(f"Spawned total {len(workers)} workers.")
    return workers
