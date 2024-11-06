import math
from functools import partial
from torch.optim.lr_scheduler import LambdaLR

def _get_wsd_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
):
    if current_step < num_warmup_steps:
        # Warmup phase: increase linearly
        return float(current_step) / float(max(1, num_warmup_steps))
    elif current_step < num_warmup_steps + num_stable_steps:
        # Stable phase: keep learning rate constant
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
):
    lr_lambda = partial(
        _get_wsd_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=num_decay_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)