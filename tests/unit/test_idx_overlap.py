from types import SimpleNamespace

import torch
import pytest

# ---- import the function under test -----------------------------------------
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
    # single parameter called "w", concatenate chunks along new dim
    # shape: (P, *chunk_dims, k)
    state_dict.widxs = [
        torch.stack(idx_lists[p], dim=0)
        if isinstance(idx_lists[p], list)
        else idx_lists[p]
        for p in range(P)
    ]

    gather_result = SimpleNamespace(uids=uids, state_dict=state_dict)
    neuron = SimpleNamespace(
        comms=_FakeComms(ts_map),
        model=_DummyModel(),  # .named_parameters()->[('w', …)]
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
    same_idxs = [torch.tensor([0, 1, 2], dtype=torch.int16)]
    idx_lists = [same_idxs, same_idxs]  # P=2, C=1, k=3
    ts_map = {0: 100, 1: 200}  # peer 1 uploaded later
    neuron, gather = _make_inputs(idx_lists, ts_map)

    res = await check_uid_index_overlap(neuron, gather, window=1)
    assert res["pairs_checked"] == 1
    assert res["pairs_high_ovlap"] == 1
    assert res["uids_over_thresh"] == {1}  # offender is the later one
    assert pytest.approx(res["mean_overlap"]) == 1.0


@pytest.mark.asyncio
async def test_detect_no_overlap():
    # Two peers with disjoint indices -> 0 % overlap (below threshold)
    idx_lists = [
        [torch.tensor([0, 1, 2], dtype=torch.int16)],
        [torch.tensor([3, 4, 5], dtype=torch.int16)],
    ]
    ts_map = {0: 100, 1: 101}
    neuron, gather = _make_inputs(idx_lists, ts_map)

    res = await check_uid_index_overlap(neuron, gather, window=1)
    assert res["pairs_high_ovlap"] == 0
    assert res["uids_over_thresh"] == set()
    assert pytest.approx(res["mean_overlap"]) == 0.0
