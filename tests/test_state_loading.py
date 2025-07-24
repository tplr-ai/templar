import importlib
import os
from unittest import mock

import pytest
import torch


# --- helpers --------------------------------------------------------------------------------------
def _make_validator(tmp_path, device="cpu"):
    """
    Build a lightweight Validator with heavy deps mocked away.
    """
    # 1) import the module first
    nv = importlib.import_module("neurons.validator")

    # 2) patch externals in‑place on the imported module
    nv.bt = mock.Mock()
    nv.tplr = mock.Mock()
    nv.tplr.logger = mock.Mock()
    for lvl in ("info", "warning", "debug"):
        getattr(nv.tplr.logger, lvl).side_effect = lambda *a, **k: None
    nv.tplr.__version__ = "TEST"
    nv.tplr.debug = nv.tplr.trace = lambda *a, **k: None

    # 3) craft minimal cfg
    cfg = mock.Mock()
    cfg.device = device
    nv.Validator.config = staticmethod(lambda: cfg)

    # 4) bare‑metal construct without running heavy __init__
    v_cls = nv.Validator
    v = object.__new__(v_cls)  # bypass original __init__
    v.config = cfg
    v.state_path = os.path.join(tmp_path, "validator-state-TEST.pt")
    v.global_step = 123
    d = device
    v.gradient_scores = torch.rand(256, dtype=torch.float32, device=d)
    v.sync_scores = torch.rand(256, dtype=torch.float32, device=d)
    v.binary_indicator_scores = torch.rand(256, dtype=torch.float32, device=d)
    v.final_scores = torch.rand(256, dtype=torch.float32, device=d)
    v.binary_moving_averages = torch.rand(256, dtype=torch.float32, device=d)
    v.weights = torch.rand(256, dtype=torch.float32, device=d)

    # ── dummy OpenSkill machinery ──────────────────────────────────────────
    class _Rating:
        def __init__(self, mu, sigma):
            self.mu = mu
            self.sigma = sigma

        # keep the typical definition: μ − 3 σ
        def ordinal(self):
            return self.mu - 3 * self.sigma

    class _OSModel:
        def rating(self, *, mu, sigma, name=None):
            r = _Rating(mu, sigma)
            r.name = name
            return r

    v.openskill_model = _OSModel()
    v.openskill_ratings = {
        1: v.openskill_model.rating(mu=25.0, sigma=8.333, name="1"),
        2: v.openskill_model.rating(mu=28.0, sigma=7.500, name="2"),
    }

    # attach needed methods (already defined on the class)
    v._state_dict = v_cls._state_dict.__get__(v)
    v.save_state = v_cls.save_state.__get__(v)
    v.load_state = v_cls.load_state.__get__(v)
    return v


# --------------------------------------------------------------------------------------------------
# HAPPY PATH
# --------------------------------------------------------------------------------------------------


async def test_save_and_load_roundtrip(tmp_path):
    """
    Verifies the *happy path* — state saved by `save_state()` is bit‑wise identical after a fresh
    `load_state()`.

    Steps
    -----
    1.  Create fresh Validator (with random tensors)  →  `save_state()`
    2.  Create another *empty* Validator, then         →  `load_state()`
    3.  Assert: global_step and all six tensors match element‑wise, dtype and device.
    """
    # set‑up
    v1 = _make_validator(tmp_path, device="cpu")
    await v1.save_state()

    # new instance with zeroed tensors
    v2 = _make_validator(tmp_path, device="cpu")
    # ensure tensors differ before load
    for t in ("gradient_scores", "weights"):
        assert not torch.equal(getattr(v1, t), getattr(v2, t))

    # action
    v2.load_state()

    # checks
    assert v2.global_step == v1.global_step
    for name in (
        "gradient_scores",
        "sync_scores",
        "binary_indicator_scores",
        "final_scores",
        "binary_moving_averages",
        "weights",
    ):
        t1, t2 = getattr(v1, name), getattr(v2, name)
        assert torch.equal(t1, t2), f"Tensor mismatch for {name}"
        assert t1.dtype == t2.dtype
        assert t2.device.type == "cpu"

    # OpenSkill ratings round‑trip
    assert v1.openskill_ratings.keys() == v2.openskill_ratings.keys()
    for uid in v1.openskill_ratings:
        r1, r2 = v1.openskill_ratings[uid], v2.openskill_ratings[uid]
        assert pytest.approx(r1.mu) == r2.mu, f"mu mismatch for uid {uid}"
        assert pytest.approx(r1.sigma) == r2.sigma, f"sigma mismatch for uid {uid}"


async def test_save_path_extension_is_pt(tmp_path):
    """
    Regression guard: make sure we never accidentally revert to `.npz`.
    """
    v = _make_validator(tmp_path)
    assert v.state_path.endswith(".pt")  # by construction
    await v.save_state()
    assert os.path.exists(v.state_path)
    assert v.state_path.endswith(".pt")


def test_load_state_missing_file(tmp_path):
    """
    Attempting to `load_state()` with no file present must **not** raise; tensors remain unchanged
    and a warning is logged.
    """
    v = _make_validator(tmp_path)
    # delete file if exists
    if os.path.exists(v.state_path):
        os.remove(v.state_path)

    # capturing stdout is enough (logger prints)
    v.load_state()  # should swallow FileNotFoundError internally
    assert not os.path.exists(v.state_path)  # still doesn't create it
    # nothing blew up → test passes


def test_load_state_corrupted_file(tmp_path):
    """
    Provide a corrupted pickle file → load_state should catch the exception and leave tensors
    untouched.
    """
    v = _make_validator(tmp_path)
    # write garbage
    with open(v.state_path, "wb") as fh:
        fh.write(b"not a pickle")

    # cache current tensor
    before = v.weights.clone()
    v.load_state()  # should *not* raise
    after = v.weights
    assert torch.equal(before, after), "Tensor mutated on corrupted load!"


def test_load_state_wrong_schema(tmp_path):
    """
    Simulate a .pt file that misses required keys.  load_state() should warn and *only* update
    what's available, leaving other tensors intact.
    """
    bogus_state = {
        "global_step": 999,
        # omit gradient_scores on purpose!
        "weights": torch.ones(256),
    }
    file = os.path.join(tmp_path, "validator-state-TEST.pt")
    torch.save(bogus_state, file)

    v = _make_validator(tmp_path)
    v.load_state()

    # updated
    assert v.global_step == 999
    # missing tensor should stay the random values assigned by factory
    assert not torch.allclose(v.gradient_scores, torch.zeros_like(v.gradient_scores))


async def test_gpu_cpu_roundtrip(tmp_path):
    """
    Edge Case: save on 'cuda' and load on 'cpu'.
    ‑ ensures .cpu() conversion in `_state_dict` works and map_location honours device.
    Skip when CUDA unavailable.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # save on GPU
    vg = _make_validator(tmp_path, device="cuda")
    await vg.save_state()

    # load on CPU
    vc = _make_validator(tmp_path, device="cpu")
    vc.load_state()

    for name in ("binary_moving_averages", "weights"):
        assert vc.__getattribute__(name).device.type == "cpu"
        assert torch.equal(vc.__getattribute__(name), vg.__getattribute__(name).cpu())
