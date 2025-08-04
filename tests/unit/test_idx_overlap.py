from types import SimpleNamespace

import pytest
import torch

# ---- import the function under test -----------------------------------------
from tplr.compress import pack_12bit_indices
from tplr.neurons import check_uid_index_overlap  # <-- fix this path

# -----------------------------------------------------------------------------


class _FakeComms:
    """Minimal stub for neuron.comms that just returns preset timestamps."""

    def __init__(self, ts_map):
        self._ts = ts_map

    async def gradient_timestamp(self, uid, _):
        return self._ts[uid]


class _DummyModel(torch.nn.Module):
    """Model with one named parameter; value is irrelevant."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(1))


def _make_inputs(idx_lists, ts_map):
    """
    Helper to build (neuron, gather_result, window).

    Parameters
    ----------
    idx_lists : list[list[torch.Tensor]]
        idx_lists[p][c] is the index tensor sent by peer *p* for chunk *c*.
        All tensors must share shape (..., k).
    ts_map    : dict[int, int]
        uid -> fake timestamp (higher == later).
    """
    uids = list(ts_map)
    P = len(uids)

    # gather_result.state_dict expects attribute <param_name + "idxs">
    state_dict = SimpleNamespace()
    # Convert to 12-bit packed format for each peer
    packed_indices = []
    vals_list = []  # Need vals for shape information
    for p in range(P):
        if isinstance(idx_lists[p], list):
            # Stack chunks and pack
            stacked = torch.stack(idx_lists[p], dim=0)
            packed_data = pack_12bit_indices(stacked)
            packed_indices.append(packed_data)
            # Create dummy values with same shape as indices
            vals_list.append(torch.randn_like(stacked, dtype=torch.float32))
        else:
            # Single tensor, pack directly
            packed_data = pack_12bit_indices(idx_lists[p])
            packed_indices.append(packed_data)
            # Create dummy values with same shape as indices
            vals_list.append(torch.randn_like(idx_lists[p], dtype=torch.float32))

    state_dict.widxs = packed_indices
    state_dict.wvals = vals_list  # Add vals for unpacking

    gather_result = SimpleNamespace(uids=uids, state_dict=state_dict)
    neuron = SimpleNamespace(
        comms=_FakeComms(ts_map),
        model=_DummyModel(),  # .named_parameters()->[('w', …)]
        config=SimpleNamespace(device="cpu"),  # Add device config
    )
    return neuron, gather_result


# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_skip_when_less_than_two():
    neuron, gather = _make_inputs([], {})  # zero peers
    res = await check_uid_index_overlap(neuron, gather, window=1)
    assert res["pairs_checked"] == res["pairs_high_ovlap"] == 0
    assert res["mean_overlap"] == 0.0 and res["uids_over_thresh"] == {}


@pytest.mark.asyncio
async def test_detect_full_overlap_and_offender_choice():
    # Two peers share exactly the same indices -> 100 % overlap (over threshold)
    same_idxs = [
        torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    ]  # Even count for 12-bit packing
    idx_lists = [same_idxs, same_idxs]  # P=2, C=1, k=3
    ts_map = {0: 100, 1: 200}  # peer 1 uploaded later
    neuron, gather = _make_inputs(idx_lists, ts_map)

    res = await check_uid_index_overlap(neuron, gather, window=1)
    assert res["pairs_checked"] == 1
    assert res["pairs_high_ovlap"] == 1
    assert res["uids_over_thresh"] == {1: "mega"}  # offender is the later one
    assert pytest.approx(res["mean_overlap"]) == 1.0


@pytest.mark.asyncio
async def test_detect_no_overlap():
    # Two peers with disjoint indices -> 0 % overlap (below threshold)
    idx_lists = [
        [torch.tensor([0, 1, 2, 3], dtype=torch.int64)],  # Even count
        [torch.tensor([4, 5, 6, 7], dtype=torch.int64)],  # Even count
    ]
    ts_map = {0: 100, 1: 101}
    neuron, gather = _make_inputs(idx_lists, ts_map)

    res = await check_uid_index_overlap(neuron, gather, window=1)
    assert res["pairs_high_ovlap"] == 0
    assert res["uids_over_thresh"] == {}
    assert pytest.approx(res["mean_overlap"]) == 0.0


# -----------------------------------------------------------------------------
# ADDITIONAL TEST CASES FOR BETTER COVERAGE
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_partial_overlap_50_percent():
    """Test 50% overlap between two peers"""
    idx_lists = [
        [torch.tensor([0, 1, 2, 3], dtype=torch.int64)],
        [torch.tensor([2, 3, 4, 5], dtype=torch.int64)],  # 50% overlap (2,3 shared)
    ]
    ts_map = {0: 100, 1: 200}
    neuron, gather = _make_inputs(idx_lists, ts_map)

    res = await check_uid_index_overlap(neuron, gather, window=1)
    assert res["pairs_checked"] == 1
    assert res["pairs_high_ovlap"] == 0  # 50% is below default 90% threshold
    assert pytest.approx(res["mean_overlap"], rel=0.01) == 0.5
    assert res["uids_over_thresh"] == {}


@pytest.mark.asyncio
async def test_custom_threshold_50_percent():
    """Test with custom 50% threshold"""
    idx_lists = [
        [torch.tensor([0, 1, 2, 3], dtype=torch.int64)],
        [torch.tensor([2, 3, 4, 5], dtype=torch.int64)],  # 50% overlap
    ]
    ts_map = {0: 100, 1: 200}
    neuron, gather = _make_inputs(idx_lists, ts_map)

    res = await check_uid_index_overlap(neuron, gather, window=1, overlap_threshold=0.5)
    assert res["pairs_checked"] == 1
    assert res["pairs_high_ovlap"] == 1  # Now meets 50% threshold
    assert 1 in res["uids_over_thresh"]  # Peer 1 is the offender (later timestamp)
    assert res["uids_over_thresh"][1] == "max"  # 50% overlap gets "max" severity


@pytest.mark.asyncio
async def test_three_peers_varying_overlap():
    """Test three peers with different overlap patterns"""
    idx_lists = [
        [torch.tensor([0, 1, 2, 3], dtype=torch.int64)],  # Peer 0
        [torch.tensor([2, 3, 4, 5], dtype=torch.int64)],  # Peer 1: 50% with 0
        [
            torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        ],  # Peer 2: 100% with 0, 50% with 1
    ]
    ts_map = {0: 100, 1: 200, 2: 300}
    neuron, gather = _make_inputs(idx_lists, ts_map)

    res = await check_uid_index_overlap(neuron, gather, window=1)
    assert res["pairs_checked"] == 3  # (0,1), (0,2), (1,2)
    assert res["pairs_high_ovlap"] == 1  # Only (0,2) meets 90% threshold
    assert 2 in res["uids_over_thresh"]  # Peer 2 is offender (latest timestamp)

    # Verify min/max tracking
    assert res["min_overlap"] == pytest.approx(0.5, rel=0.01)  # Min is 50%
    assert res["max_overlap"] == pytest.approx(1.0, rel=0.01)  # Max is 100%


@pytest.mark.asyncio
async def test_multiple_chunks_per_peer():
    """Test with multiple chunks per peer"""
    idx_lists = [
        # Peer 0: 2 chunks
        [
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([2, 3], dtype=torch.int64),
        ],
        # Peer 1: 2 chunks with overlap
        [
            torch.tensor([0, 1], dtype=torch.int64),  # 100% overlap with peer 0 chunk 0
            torch.tensor([4, 5], dtype=torch.int64),  # 0% overlap with peer 0 chunk 1
        ],
    ]
    ts_map = {0: 100, 1: 200}
    neuron, gather = _make_inputs(idx_lists, ts_map)

    res = await check_uid_index_overlap(neuron, gather, window=1)
    assert res["pairs_checked"] == 1
    # Average overlap across chunks: (100% + 0%) / 2 = 50%
    assert pytest.approx(res["mean_overlap"], rel=0.01) == 0.5
    assert res["pairs_high_ovlap"] == 0  # 50% average doesn't meet threshold


@pytest.mark.asyncio
async def test_multiple_chunks_high_overlap():
    """Test multiple chunks with consistently high overlap"""
    idx_lists = [
        # Peer 0: 2 chunks
        [
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([2, 3], dtype=torch.int64),
        ],
        # Peer 1: 2 chunks with high overlap
        [
            torch.tensor([0, 1], dtype=torch.int64),  # 100% overlap
            torch.tensor([2, 3], dtype=torch.int64),  # 100% overlap
        ],
    ]
    ts_map = {0: 100, 1: 200}
    neuron, gather = _make_inputs(idx_lists, ts_map)

    res = await check_uid_index_overlap(neuron, gather, window=1)
    assert res["pairs_checked"] == 1
    assert res["pairs_high_ovlap"] == 1  # 100% average meets threshold
    assert 1 in res["uids_over_thresh"]
    assert res["uids_over_thresh"][1] == "mega"  # 100% overlap gets "mega" severity


@pytest.mark.asyncio
async def test_large_indices_range():
    """Test with large index values within 12-bit range"""
    idx_lists = [
        [torch.tensor([4000, 4001, 4002, 4003], dtype=torch.int64)],
        [torch.tensor([4002, 4003, 4004, 4005], dtype=torch.int64)],  # 50% overlap
    ]
    ts_map = {0: 100, 1: 200}
    neuron, gather = _make_inputs(idx_lists, ts_map)

    res = await check_uid_index_overlap(neuron, gather, window=1)
    assert res["pairs_checked"] == 1
    assert pytest.approx(res["mean_overlap"], rel=0.01) == 0.5


@pytest.mark.asyncio
async def test_multiple_offenders():
    """Test when multiple pairs exceed threshold"""
    idx_lists = [
        [torch.tensor([0, 1, 2, 3], dtype=torch.int64)],  # Peer 0
        [torch.tensor([0, 1, 2, 3], dtype=torch.int64)],  # Peer 1: 100% with 0
        [torch.tensor([0, 1, 2, 3], dtype=torch.int64)],  # Peer 2: 100% with 0 and 1
    ]
    ts_map = {0: 100, 1: 200, 2: 300}
    neuron, gather = _make_inputs(idx_lists, ts_map)

    res = await check_uid_index_overlap(neuron, gather, window=1)
    assert res["pairs_checked"] == 3
    assert res["pairs_high_ovlap"] == 3  # All pairs have 100% overlap
    # Peers 1 and 2 should both be offenders
    assert set(res["uids_over_thresh"].keys()) == {1, 2}
    assert res["uids_over_thresh"][1] == "mega"  # 100% overlap
    assert res["uids_over_thresh"][2] == "mega"  # 100% overlap


@pytest.mark.asyncio
async def test_four_peers_complex_pattern():
    """Test four peers with complex overlap patterns"""
    idx_lists = [
        [torch.tensor([0, 1, 2, 3], dtype=torch.int64)],  # Peer 0
        [torch.tensor([4, 5, 6, 7], dtype=torch.int64)],  # Peer 1: no overlap with 0
        [
            torch.tensor([0, 1, 6, 7], dtype=torch.int64)
        ],  # Peer 2: 50% with 0, 50% with 1
        [
            torch.tensor([2, 3, 4, 5], dtype=torch.int64)
        ],  # Peer 3: 50% with 0, 50% with 1
    ]
    ts_map = {0: 100, 1: 200, 2: 300, 3: 400}
    neuron, gather = _make_inputs(idx_lists, ts_map)

    res = await check_uid_index_overlap(neuron, gather, window=1)
    assert res["pairs_checked"] == 6  # C(4,2) = 6 pairs
    assert res["pairs_high_ovlap"] == 0  # No pair meets 90% threshold
    assert res["uids_over_thresh"] == {}


@pytest.mark.asyncio
async def test_empty_indices():
    """Test with empty index tensors"""
    idx_lists = [
        [torch.tensor([], dtype=torch.int64).reshape(0)],  # Empty with shape (0,)
        [torch.tensor([], dtype=torch.int64).reshape(0)],  # Empty with shape (0,)
    ]
    ts_map = {0: 100, 1: 200}
    neuron, gather = _make_inputs(idx_lists, ts_map)

    res = await check_uid_index_overlap(neuron, gather, window=1)
    assert res["pairs_checked"] == 1
    # Empty tensors can't overlap
    assert res["mean_overlap"] == 0.0


@pytest.mark.asyncio
async def test_offender_selection_same_timestamp():
    """Test offender selection when timestamps are identical"""
    idx_lists = [
        [torch.tensor([0, 1, 2, 3], dtype=torch.int64)],
        [torch.tensor([0, 1, 2, 3], dtype=torch.int64)],  # 100% overlap
    ]
    ts_map = {0: 100, 1: 100}  # Same timestamp
    neuron, gather = _make_inputs(idx_lists, ts_map)

    res = await check_uid_index_overlap(neuron, gather, window=1)
    assert res["pairs_checked"] == 1
    assert res["pairs_high_ovlap"] == 1
    # When timestamps are equal, either could be selected (implementation dependent)
    assert len(res["uids_over_thresh"]) == 1
    assert set(res["uids_over_thresh"].keys()).issubset({0, 1})
    # Check the severity is mega for 100% overlap
    offender_uid = list(res["uids_over_thresh"].keys())[0]
    assert res["uids_over_thresh"][offender_uid] == "mega"
