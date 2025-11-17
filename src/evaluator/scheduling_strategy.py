"""
Defines the scheduling strategies for the ParallelEvaluator.

This module uses the Strategy design pattern to encapsulate different ways of
distributing the evaluation workload (e.g., CPU only, GPU only, Hybrid).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

import torch
import torch.multiprocessing as mp
import time

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
        log_queue: Any = None,
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

        return _spawn_processes(
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

        return _spawn_processes(
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

        return _spawn_processes(
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
            log_queue=kwargs.get("log_queue", None),
            session_log_filename=kwargs["session_log_filename"],
            dl_workers_per_gpu=dl_per_gpu,
            fixed_batch_size=kwargs.get("fixed_batch_size")
        )

        cpu_workers = _spawn_processes(
            ctx=ctx,
            task_queue=cpu_task_queue,
            result_queue=kwargs["result_queue"],
            log_queue=kwargs.get("log_queue", None),
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

def _sspawn_processes(
    ctx: Any,
    result_queue: mp.Queue,
    session_log_filename: str,
    task_queue: mp.Queue,
    log_queue: Any = None, 
    num_gpu_workers: int = 0,
    num_cpu_workers: int = 0,
    dl_workers_per_gpu: int = 1,
    dl_workers_per_cpu: int = 1,
    fixed_batch_size: Optional[int] = None,
    gpu_worker_offset: int = 0,
) -> List[mp.Process]:
    """Helper to spawn and start worker processes."""
    workers = []


    for i in range(num_cpu_workers):
        w_config = WorkerConfig(
            worker_id=i, 
            device="cpu",
            task_queue=task_queue,
            result_queue=result_queue,
            log_queue=log_queue,
            session_log_filename=session_log_filename,
            num_dataloader_workers=dl_workers_per_cpu,
            fixed_batch_size=fixed_batch_size,
            total_cpu_workers=num_cpu_workers,
        )
        p = ctx.Process(target=worker_main, args=(w_config,))
        p.start()
        workers.append(p)
    
    # Przesunięcie dla procesów GPU będzie równe liczbie procesów CPU
    gpu_worker_start_id = num_cpu_workers
    time.sleep(1.0) # Zachowujemy opóźnienie, by dać CPU czas na inicjalizację

    # 2. Spawn GPU workers (NOWA KOLEJNOŚĆ)
    for i in range(num_gpu_workers):
        w_config = WorkerConfig(
            # ID procesów GPU zaczynają się od ID pierwszego worker-a CPU + num_cpu_workers
            worker_id=i + gpu_worker_start_id,
            device=i,
            task_queue=task_queue,
            result_queue=result_queue,
            session_log_filename=session_log_filename,
            log_queue=log_queue,
            num_dataloader_workers=dl_workers_per_gpu,
            fixed_batch_size=fixed_batch_size,
        )
        p = ctx.Process(target=worker_main, args=(w_config,))
        p.start()
        workers.append(p)

    return workers

def _spawn_processes(
    ctx: Any, # Multiprocessing context (e.g., mp.get_context('spawn'))
    result_queue: mp.Queue,
    session_log_filename: str,
    task_queue: mp.Queue, # Single task queue (used by CPUOnly, GPUOnly, SimpleHybrid)
                          # Or the specific queue (GPU/CPU) when called by HybridStrategy
    log_queue: Any = None,
    num_gpu_workers: int = 0,
    num_cpu_workers: int = 0,
    dl_workers_per_gpu: int = 1,
    dl_workers_per_cpu: int = 1,
    fixed_batch_size: Optional[int] = None,
    gpu_worker_offset: int = 0, # Offset for GPU worker IDs (e.g., num_cpu_workers)
) -> List[mp.Process]:
    """
    Helper to spawn GPU and/or CPU workers using the provided task queue.

    Implements NUMA-aware core pinning based on the 64-core system:
    - CPU Workers: 12 cores each, assigned to the first available cores on a NUMA node.
    - GPU Workers: 4 cores each, assigned to the last available cores on their GPU's NUMA node.
    Handles cases where num_cpu_workers or num_gpu_workers is 0.
    """
    workers = [] # List to hold the spawned process objects

    if num_gpu_workers <= 0 and num_cpu_workers <= 0:
        return workers # Return empty list, no failure

    # --- NUMA Configuration Based on Your System ---
    # GPU Device ID -> NUMA Node ID (from nvidia-smi topo -m)
    gpu_to_node_map = {0: 1, 1: 0, 2: 3, 3: 2}

    # Core IDs for each NUMA node (16 cores each, no Hyper-threading)
    node_cores = {
        0: list(range(0, 16)),  # NUMA Node 0: Cores 0-15
        1: list(range(16, 32)), # NUMA Node 1: Cores 16-31
        2: list(range(32, 48)), # NUMA Node 2: Cores 32-47
        3: list(range(48, 64)), # NUMA Node 3: Cores 48-63
    }

    cores_per_cpu_worker = 14
    cores_per_gpu_worker = 2
    total_cores_available = 64
    # --- End NUMA Configuration ---

    # --- Core Assignment Tracking (Local to this function call) ---
    # We assume the calling strategy ensures non-overlapping calls if managing a global pool.
    # For simplicity here, we track within this function's scope.
    # A more robust global tracker might be needed if _spawn_processes could be called concurrently.
    assigned_core_mask = [False] * total_cores_available
    # --- End Core Assignment Tracking ---


    # --- 1. Spawn GPU Workers (If requested in this call) ---
    if num_gpu_workers > 0:
        logger.info(f"Spawning {num_gpu_workers} GPU workers ({cores_per_gpu_worker} cores each, NUMA-aware)...")
        for device_id in range(num_gpu_workers):
            if device_id not in gpu_to_node_map:
                 logger.error(f"Missing NUMA mapping for GPU device ID {device_id}. Skipping.")
                 continue
            node_id = gpu_to_node_map[device_id]
            if node_id not in node_cores or len(node_cores[node_id]) < cores_per_gpu_worker:
                 logger.error(f"NUMA Node {node_id} invalid/insufficient cores for GPU {device_id}.")
                 continue
            core_ids = node_cores[node_id][-cores_per_gpu_worker:]
            conflict = False
            for core in core_ids:
                if not (0 <= core < total_cores_available) or assigned_core_mask[core]:
                    logger.error(f"Core conflict/invalid core {core} for GPU {device_id} on Node {node_id}.")
                    conflict = True
                    break
                assigned_core_mask[core] = True # Mark core assigned *within this call*
            if conflict: continue

            worker_id = device_id + gpu_worker_offset # Apply offset for unique ID
            logger.info(f"  Spawning Worker-{worker_id} (GPU device={device_id}) -> Node {node_id} (Cores {core_ids[0]}-{core_ids[-1]})")
            w_config = WorkerConfig(
                worker_id=worker_id, device=device_id, core_ids=core_ids,
                task_queue=task_queue, result_queue=result_queue, log_queue=log_queue,
                session_log_filename=session_log_filename,
                num_dataloader_workers=dl_workers_per_gpu,
                fixed_batch_size=fixed_batch_size,
            )
            p = ctx.Process(target=worker_main, args=(w_config,))
            p.start()
            workers.append(p)
        # Optional delay only if GPUs were spawned in this call
        time.sleep(1.0)


    # --- 2. Spawn CPU Workers (If requested in this call) ---
    if num_cpu_workers > 0:
        logger.info(f"Spawning {num_cpu_workers} CPU workers ({cores_per_cpu_worker} cores each, NUMA-aware)...")
        cpu_node_assignment_order = sorted(list(node_cores.keys()))
        for i in range(num_cpu_workers):
            node_id = cpu_node_assignment_order[i % len(cpu_node_assignment_order)]
            if node_id not in node_cores or len(node_cores[node_id]) < cores_per_cpu_worker:
                 logger.error(f"NUMA Node {node_id} invalid/insufficient cores for CPU {i}.")
                 continue
            core_ids = node_cores[node_id][:cores_per_cpu_worker]
            conflict = False
            for core in core_ids:
                # Check against cores potentially assigned to GPUs *in this same function call*
                if not (0 <= core < total_cores_available) or assigned_core_mask[core]:
                     logger.error(f"Core conflict/invalid core {core} for CPU {i} on Node {node_id}.")
                     conflict = True
                     break
                assigned_core_mask[core] = True # Mark core assigned *within this call*
            if conflict: continue

            worker_id = i # CPU worker IDs start from 0 relative to their group
                          # The calling strategy (like HybridStrategy) manages global uniqueness
            logger.info(f"  Spawning Worker-{worker_id} (CPU) -> Node {node_id} (Cores {core_ids[0]}-{core_ids[-1]})")
            w_config = WorkerConfig(
                worker_id=worker_id, device="cpu", core_ids=core_ids,
                task_queue=task_queue, result_queue=result_queue, log_queue=log_queue,
                session_log_filename=session_log_filename,
                num_dataloader_workers=dl_workers_per_cpu,
                fixed_batch_size=fixed_batch_size,
                total_cpu_workers=num_cpu_workers # Pass if worker needs this info
            )
            p = ctx.Process(target=worker_main, args=(w_config,))
            p.start()
            workers.append(p)

    # --- Final Logging for this call ---
    # Sanity check for unassigned cores *within the scope of this call* is less useful
    # as it depends on whether both CPU and GPU workers were spawned together.
    logger.info(f"Finished spawning group. Workers created in this call: {len(workers)}.")

    return workers # Return list of processes spawned *in this call*
