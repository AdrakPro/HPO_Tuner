import numpy as np
import torch


def mixup_data(x, y, alpha=0.2, device="cuda"):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_schedule(
    epoch: int,
    total_epochs: int,
    base_alpha: float,
    warmup_frac: float = 0.15,
    cooldown_frac: float = 0.15,
) -> float:
    warmup_e = int(total_epochs * warmup_frac)
    cooldown_e = int(total_epochs * cooldown_frac)
    hold_end = total_epochs - cooldown_e

    if epoch < warmup_e:
        # Linear ramp from 0 -> base_alpha
        return base_alpha * (epoch / max(1, warmup_e))
    elif epoch < hold_end:
        # Flat at base_alpha
        return base_alpha
    else:
        # Off in the last phase
        return 0.0
