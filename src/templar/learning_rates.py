# Global imports
import math
from functools import partial
from torch.optim.lr_scheduler import LambdaLR


def _get_wsd_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
) -> float:
    """
    Calculates the learning rate multiplier for Warmup-Stable-Decay (WSD) schedule.

    The schedule consists of three phases:
    1. Warmup: Linear increase from 0 to 1 over num_warmup_steps
    2. Stable: Constant learning rate of 1.0 for num_stable_steps
    3. Decay: Square root decay from 1.0 to 0.0 over num_decay_steps

    Args:
        current_step (int): Current training step
        num_warmup_steps (int): Number of warmup steps
        num_stable_steps (int): Number of steps at constant learning rate
        num_decay_steps (int): Number of decay steps

    Returns:
        float: Learning rate multiplier between 0.0 and 1.0
    """
    if current_step < num_warmup_steps:
        # Warmup phase: increase linearly from 0 to 1
        return float(current_step) / float(max(1, num_warmup_steps))
    elif current_step < num_warmup_steps + num_stable_steps:
        # Stable phase: keep learning rate constant at 1.0
        return 1.0
    else:
        # Decay phase: decrease following a 1 - sqrt(x) schedule
        decay_step = current_step - num_warmup_steps - num_stable_steps
        decay_progress = float(decay_step) / float(max(1, num_decay_steps))
        return max(0.0, 1 - math.sqrt(decay_progress))


def get_wsd_scheduler(
    optimizer,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Creates a learning rate scheduler with Warmup-Stable-Decay schedule.

    This scheduler adjusts the learning rate according to three phases:
    1. Linear warmup for num_warmup_steps
    2. Constant learning rate for num_stable_steps
    3. Square root decay for num_decay_steps

    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps (int): Number of warmup steps
        num_stable_steps (int): Number of steps at constant learning rate
        num_decay_steps (int): Number of decay steps
        last_epoch (int, optional): The index of last epoch. Default: -1

    Returns:
        LambdaLR: PyTorch learning rate scheduler

    Example:
        >>> optimizer = AdamW(model.parameters(), lr=1e-3)
        >>> scheduler = get_wsd_scheduler(
        ...     optimizer,
        ...     num_warmup_steps=1000,
        ...     num_stable_steps=10000,
        ...     num_decay_steps=5000
        ... )
    """
    lr_lambda = partial(
        _get_wsd_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=num_decay_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
