import numpy as np
from torch.utils.data import Sampler

__all__ = ["MinerSampler", "EvalSampler"]


def _window_seed(uid: int, window: int) -> int:
    """Deterministic 32-bit seed for (uid, window)."""
    return (uid * 1_000_003 ^ window) & 0xFFFF_FFFF


# ────────────────────────────────────────────────────────────────────────────
# MinerSampler
# ────────────────────────────────────────────────────────────────────────────
class MinerSampler(Sampler):
    """
    Deterministic, rank-aware sampler for training **miners**.

    Arguments
    ---------
    dataset_len   : total number of token-sequences in the dataset
    uid, window   : miner id & current window number (deterministic seed)
    steps_per_window (H)
    micro_bs      : per-GPU micro-batch size (tokens sent to forward())
    batch_size    : *global* batch size per optimizer step
    rank, world_size
    """

    def __init__(
        self,
        dataset_len: int,
        uid: int,
        window: int,
        steps_per_window: int,   # H
        micro_bs: int,           # per-GPU micro-batch
        batch_size: int,         # global batch (all GPUs × grad_accum)
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset_len = dataset_len
        self.steps_per_window = steps_per_window
        self.micro_bs = micro_bs
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size

        # ── derive grad-accumulation factor ────────────────────────────────
        denom = micro_bs * world_size
        if batch_size % denom != 0:
            raise ValueError(
                f"batch_size={batch_size} is not divisible by "
                f"micro_bs·world_size = {denom}"
            )
        self.grad_accum = batch_size // denom

        # Initialize with given uid and window
        self.set_window_uid(uid, window)

    def set_window_uid(self, uid: int, window: int):
        """Update the sampler for a new window/uid combination."""
        self.uid = uid
        self.window = window
        
        # ── global number of samples in this training window ───────────────
        wanted_global = self.steps_per_window * self.batch_size
        if wanted_global > self.dataset_len:
            raise ValueError(
                f"Window needs {wanted_global} samples but dataset has "
                f"{self.dataset_len}"
            )

        # ── deterministic choice & rank slicing ────────────────────────────
        rng = np.random.default_rng(_window_seed(uid, window))
        full = rng.choice(self.dataset_len, size=wanted_global, replace=False)
        self._local = full[self.rank :: self.world_size].tolist()

    # Sampler API
    def __iter__(self):
        return iter(self._local)

    def __len__(self):
        # = H × grad_accum × micro_bs  (per GPU)
        return len(self._local)


# ────────────────────────────────────────────────────────────────────────────
# EvalSampler
# ────────────────────────────────────────────────────────────────────────────
class EvalSampler(Sampler):
    """
    Deterministic sampler for **validators** with possibly multiple GPUs.

    The validator reproduces the *training* pool first
    (size = H × batch_size) and then selects a smaller,
    user-defined subset of size H × validation_bs.

    validation_bs ≤ batch_size.

    The resulting indices are again split evenly across ranks.
    """

    def __init__(
        self,
        dataset_len: int,
        uid: int,
        window: int,
        steps_per_window: int,   # H
        micro_bs: int,
        batch_size: int,
        validation_bs: int,      # ≤ batch_size
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset_len = dataset_len
        self.steps_per_window = steps_per_window
        self.micro_bs = micro_bs
        self.batch_size = batch_size
        self.validation_bs = validation_bs
        self.rank = rank
        self.world_size = world_size

        if validation_bs > batch_size:
            raise ValueError("validation_bs must be ≤ batch_size")

        # ── derive grad_accum for symmetry check (not otherwise used) ──────
        denom = micro_bs * world_size
        if batch_size % denom != 0:
            raise ValueError(
                f"batch_size={batch_size} is not divisible by "
                f"micro_bs·world_size = {denom}"
            )

        # Initialize with given uid and window
        self.set_window_uid(uid, window)

    def set_window_uid(self, uid: int, window: int):
        """Update the sampler for a new window/uid combination."""
        self.uid = uid
        self.window = window
        
        # ── reconstruct the *training* pool deterministically ──────────────
        train_global = self.steps_per_window * self.batch_size
        if train_global > self.dataset_len:
            raise ValueError("Training pool larger than dataset!")

        rng_train = np.random.default_rng(_window_seed(uid, window))
        training_pool = rng_train.choice(
            self.dataset_len, size=train_global, replace=False
        )

        # ── choose validation subset from that pool ────────────────────────
        rng_val = np.random.default_rng(_window_seed(uid, window) ^ 0xA5A5_A5A5)
        val_full = rng_val.choice(training_pool, size=self.validation_bs, replace=False)

        # ── stride-split across validator GPUs ─────────────────────────────
        self._local = val_full[self.rank :: self.world_size].tolist()

    def __iter__(self):
        return iter(self._local)

    def __len__(self):
        return len(self._local)
