import pytest
import numpy as np
import tplr


# --------------------------------------------------------------------------- #
# Minimal stand-in for tplr.SharedShardedDataset
# --------------------------------------------------------------------------- #
class DummyDataset:
    def __init__(self, size: int):
        self._size = size

    def __len__(self):
        return self._size

    def sample_id(self, idx: int) -> int:  # noqa: D401
        return idx


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def dataset():
    return DummyDataset(size=10_000)


@pytest.fixture(params=[1, 2, 4])  # single-GPU plus two multi-GPU cases
def world_size(request):
    return request.param


@pytest.fixture
def common_kwargs(world_size):
    """
    Shared args.  batch_size is multiples of micro_bs * world_size so that
    grad-accumulation steps == 1 and the symmetry check passes.
    """
    return dict(
        uid=42,
        window=0,
        steps_per_window=5,
        micro_bs=8,
        batch_size=32 * world_size,
        world_size=world_size,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_eval_is_subset_of_miner(dataset, common_kwargs):
    """
    Across *all* ranks, every evaluation index must belong to the miner’s
    window-specific training pool.
    """
    ws = common_kwargs["world_size"]

    miner_union = set()

    for rank in range(ws):
        miner = tplr.MinerSampler(
            dataset,
            target_batch_size=common_kwargs["batch_size"],
            rank=rank,
            **common_kwargs,
        )
        miner_union.update(iter(miner))

    evaluator = tplr.EvalSampler(
        dataset,
        validation_bs=16,
        rank=0,
        **common_kwargs,
    )

    assert set(iter(evaluator)).issubset(miner_union), (
        "tplr.EvalSampler produced ids outside the miner’s pool"
    )
    tplr.EvalSampler._cached_indices.clear()


def test_no_duplicate_indices_in_miner(dataset, common_kwargs):
    miner = tplr.MinerSampler(
        dataset,
        target_batch_size=common_kwargs["batch_size"],
        rank=0,
        **common_kwargs,
    )
    miner_indices = list(iter(miner))
    assert len(miner_indices) == len(set(miner_indices))


def test_determinism_per_window(dataset, common_kwargs):
    miner0 = tplr.MinerSampler(
        dataset,
        target_batch_size=common_kwargs["batch_size"],
        rank=0,
        **common_kwargs,
    )
    miner1 = tplr.MinerSampler(
        dataset,
        target_batch_size=common_kwargs["batch_size"],
        rank=0,
        **common_kwargs,
    )
    assert np.array_equal(
        np.fromiter(miner0, dtype=int), np.fromiter(miner1, dtype=int)
    )


def test_world_size_partitioning(dataset):
    ws = 4
    args = dict(
        dataset=dataset,
        uid=999,
        window=3,
        steps_per_window=2,
        micro_bs=4,
        batch_size=16,
        target_batch_size=16,
        world_size=ws,
    )

    global_seen = []
    for rank in range(ws):
        sampler = tplr.MinerSampler(rank=rank, **args)
        global_seen.append(set(iter(sampler)))

    union = set().union(*global_seen)
    expected_len = args["steps_per_window"] * args["target_batch_size"]
    assert len(union) == expected_len
    # all slices must be disjoint
    for i in range(ws):
        for j in range(i + 1, ws):
            assert global_seen[i].isdisjoint(global_seen[j])


def test_dataset_too_small_raises():
    tiny = DummyDataset(size=3)
    with pytest.raises(ValueError, match="Window needs"):
        tplr.MinerSampler(
            tiny,
            uid=0,
            window=0,
            steps_per_window=2,
            micro_bs=1,
            batch_size=4,
            target_batch_size=4,
        )

    with pytest.raises(ValueError, match="Training pool larger than dataset"):
        tplr.EvalSampler(
            tiny,
            uid=0,
            window=0,
            steps_per_window=2,
            micro_bs=1,
            batch_size=4,
            validation_bs=2,
        )
