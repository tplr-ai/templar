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
    assert res["mean_overlap"] == 0.0 and res["uids_over_thresh"] == set()


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
