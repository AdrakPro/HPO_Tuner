"""
Thread optimization for PyTorch to prevent oversubscription on multi-core systems.
"""

import os
import torch
from src.logger.logger import logger


class ThreadOptimizer:
    """
    Optimizes thread usage for PyTorch and numerical libraries
    to prevent oversubscription on systems with many CPU cores.
    """

    @staticmethod
    def optimize_worker_threads(
        worker_id: int,
        total_workers: int,
        total_system_cores: int = 128,
        reserved_cores: int = 8,
    ) -> int:
        """
        Calculate and set optimal thread configuration for a worker.

        Args:
            worker_id: ID of the worker
            total_workers: Total number of workers
            total_system_cores: Total CPU cores in the system
            reserved_cores: Cores to reserve for system/GPU operations

        Returns:
            Dictionary with thread configuration
        """
        # Calculate available cores for workers
        available_cores = total_system_cores - reserved_cores
        cores_per_worker = max(1, available_cores // total_workers)

        # Limit to reasonable maximum (16 cores per worker is plenty)
        cores_per_worker = min(cores_per_worker, 16)

        logger.info(
            f"Worker {worker_id} thread optimization: {cores_per_worker} threads/worker "
            f"({total_workers} workers sharing {available_cores} cores)"
        )

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
                logger.info("TF32 enabled for A100 GPU (3x performance boost)")
            else:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                logger.info("TF32 disabled (not supported on this GPU)")

    @staticmethod
    def get_optimal_batch_size(
        device_type: str, model_parameter_count: int
    ) -> int:
        """
        Calculate optimal batch size based on device and model size.

        Args:
            device_type: 'cuda' or 'cpu'
            model_parameter_count: Number of parameters in the model

        Returns:
            Recommended batch size
        """
        if device_type == "cuda":
            # A100 can handle large batches efficiently
            if model_parameter_count < 1_000_000:  # Small model
                return 1024
            elif model_parameter_count < 10_000_000:  # Medium model
                return 512
            else:  # Large model
                return 256
        else:
            # CPU benefits from smaller batches due to memory bandwidth
            if model_parameter_count < 1_000_000:
                return 128
            elif model_parameter_count < 10_000_000:
                return 64
            else:
                return 32

    @staticmethod
    def get_system_threading_info() -> dict:
        """Get current threading configuration for diagnostics."""
        return {
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "Not set"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "Not set"),
            "pytorch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
            "tf32_enabled": (
                torch.backends.cuda.matmul.allow_tf32
                if torch.cuda.is_available()
                else False
            ),
        }
