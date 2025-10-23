"""
Thread optimization for PyTorch to prevent oversubscription on multi-core systems.
"""

import torch

class ThreadOptimizer:
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

