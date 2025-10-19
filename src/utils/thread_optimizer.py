"""
Thread optimization for PyTorch to prevent oversubscription on multi-core systems.
"""

import os
import torch
from src.logger.logger import logger
from math import floor
from typing import Tuple

class ThreadOptimizer:
    """
    Optimizes thread usage for PyTorch and numerical libraries
    to prevent oversubscription on systems with many CPU cores.
    """
    @staticmethod
    def get_total_cores() -> int:
        """Gets the total number of available CPU cores."""
        return 7 

    @staticmethod
    def calculate_worker_threads(
        total_cpu_workers: int,
        total_gpu_workers: int,
        threads_per_gpu_worker: int = 2,
    ) -> Tuple[int, int]:
        """
        Calculates the optimal number of threads for CPU and GPU workers.

        Args:
            total_cpu_workers: The number of CPU-bound worker processes.
            total_gpu_workers: The number of GPU-bound worker processes.
            threads_per_gpu_worker: The number of CPU threads to allocate per GPU worker.

        Returns:
            A tuple of (threads_for_each_cpu_worker, threads_for_each_gpu_worker).
        """
        total_cores = ThreadOptimizer.get_total_cores()
        logger.info(f"System has {total_cores} total CPU cores.")

        # Reserve a core for the OS and main process if we have enough cores
        reserved_cores = 1 if total_cores > 4 else 0

        available_cores = total_cores - reserved_cores
        cores_for_gpu_workers = total_gpu_workers * threads_per_gpu_worker

        # Ensure we don't overallocate to GPU workers
        if cores_for_gpu_workers > available_cores:
            actual_gpu_threads = max(1, floor(available_cores / total_gpu_workers) if total_gpu_workers > 0 else 0)
        else:
            actual_gpu_threads = threads_per_gpu_worker

        cores_for_gpu_workers = total_gpu_workers * actual_gpu_threads

        # Remaining cores are for CPU workers
        cpu_pool_cores = available_cores - cores_for_gpu_workers

        if total_cpu_workers > 0 and cpu_pool_cores > 0:
            cpu_worker_threads = max(1, floor(cpu_pool_cores / total_cpu_workers))
        else:
            cpu_worker_threads = 1  # Default to 1 if no cores are left or no workers

        logger.info(
            f"Thread allocation: CPU_workers={total_cpu_workers} @ {cpu_worker_threads} threads | "
            f"GPU_workers={total_gpu_workers} @ {actual_gpu_threads} threads."
        )

        return cpu_worker_threads, actual_gpu_threads

    @staticmethod
    def set_omp_threads(num_threads: int):
        """Sets the standard OpenMP/MKL environment variables for thread count."""
        str_threads = str(num_threads)
        os.environ["OMP_NUM_THREADS"] = str_threads
        os.environ["MKL_NUM_THREADS"] = str_threads
        os.environ["NUMEXPR_NUM_THREADS"] = str_threads
        os.environ["OPENBLAS_NUM_THREADS"] = str_threads
        torch.set_num_threads(num_threads)
        return cores_per_worker

    @staticmethod
    def enable_tf32():
        """Enable TF32 tensor cores for A100 GPUs (3x performance boost)."""
        if torch.cuda.is_available():
            # Check if we have A100 GPUs (Compute Capability 8.0)
            major, minor = torch.cuda.get_device_capability()
            if major >= 8:  # A100 and newer
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            else:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False

