import math

import numpy as np
import pytest
import torch

# Import from your module (adjust the import path if your file lives elsewhere)
from tplr.compress.bits import (
    BitReader,
    decode_batch_rows,
    encode_batch_rows,
    encode_batch_rows_cpu,
)

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def device_params():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


def make_even_k(k: int) -> int:
    return k if k % 2 == 0 else k - 1 if k > 0 else 0


def scatter2d(indices: torch.Tensor, values: torch.Tensor, C: int) -> torch.Tensor:
    """
    Scatter-add helper: indices [N, K], values [N, K] -> dense [N, C].
    Uses sum; for unique indices per row, mean == sum (used in your decompressor).
    """
    N, K = indices.shape
    out = torch.zeros((N, C), dtype=values.dtype, device=values.device)
    out.scatter_add_(1, indices.long(), values)
    return out


def parse_first_row_header(payload: bytes):
    """
    Read container header and the first row header to extract:
      - C, N
      - first row byte length
      - (lb, k_param, use_bitmap) for row 0
    """
    br = BitReader(payload)
    C = br.read_bits(12) + 1
    N = br.read_bits(16)
    _ = br.read_bits(1)  # reserved

    if N == 0:
        return C, N, 0, 0, 0, 0

    row_len = br.read_bits(16)
    row_bytes = br.read_bytes(row_len)
    rr = BitReader(row_bytes)
    lb = rr.read_bits(5)
    k_param = rr.read_bits(4)
    use_bitmap = rr.read_bits(1)
    return C, N, row_len, lb, k_param, use_bitmap


# -------------------------------------------------------------------------
# Core correctness tests
# -------------------------------------------------------------------------


@pytest.mark.parametrize("device", device_params())
@pytest.mark.parametrize(
    "N,C,K",
    [
        (1, 64, 6),  # C=64 is divisible by 64
        (4, 128, 8),  # C=128 is divisible by both 64 and 128
        (8, 256, 12),  # C=256 is divisible by both 64 and 128
        (3, 192, 10),  # C=192 is divisible by 64
    ],
)
def test_roundtrip_decode_matches_original_permutation(device, N, C, K):
    """
    The strongest property: for each row,
    decoded_indices == original_indices[ perm ].
    Also the sets match (ignoring ordering).
    """
    K = make_even_k(K)
    # Generate unique indices per row (as topk would produce)
    # Use a simpler approach: create ascending indices then shuffle per row
    idx = torch.zeros((N, K), device=device, dtype=torch.int64)
    for i in range(N):
        # Create a unique set of K indices for this row
        all_indices = torch.arange(C, device=device, dtype=torch.int64)
        shuffled = all_indices[torch.randperm(C, device=device)][:K]
        idx[i] = shuffled

    payload, perm, meta = encode_batch_rows(idx, C=C)  # perm: [N, K]
    rows, C2, N2 = decode_batch_rows(payload)

    assert C2 == C
    assert N2 == N
    assert perm.shape == idx.shape
    assert perm.dtype == torch.int64

    # Check permutation -> decoded indices equality
    for i in range(N):
        decoded = rows[i]
        assert len(decoded) == K
        # apply permutation (perm maps emitted-order position -> original topk position)
        perm_i = perm[i].detach().cpu().tolist()
        orig = idx[i].detach().cpu().tolist()
        reindexed = [orig[p] for p in perm_i]
        assert decoded == reindexed, f"Row {i}: decoded != idx[perm]"

        # set equality for good measure
        assert sorted(decoded) == sorted(orig)

    # meta sanity
    assert isinstance(meta, dict)
    assert "total_bits" in meta and meta["total_bits"] > 0
    assert "avg_bits_per_row" in meta and meta["avg_bits_per_row"] >= 0
    assert "B_hist" in meta and isinstance(meta["B_hist"], dict)
    assert sum(meta["B_hist"].values()) == N


@pytest.mark.parametrize("device", device_params())
def test_permutation_reorders_values_correctly(device):
    """
    If we reorder values by 'perm' and scatter into C,
    the dense reconstruction matches scattering with original (idx, values).
    """
    N, C, K = 5, 128, 8  # C=128 is divisible by both 64 and 128
    K = make_even_k(K)
    # Generate unique indices per row (as topk would produce)
    idx = torch.zeros((N, K), device=device, dtype=torch.int64)
    for i in range(N):
        idx[i] = torch.randperm(C, device=device, dtype=torch.int64)[:K]
    values = torch.randn(N, K, device=device)

    payload, perm, _ = encode_batch_rows(idx, C=C)
    rows, C2, N2 = decode_batch_rows(payload)
    assert C2 == C and N2 == N

    # original scatter
    dense_a = scatter2d(idx, values, C)

    # codec-order indices and values
    dec_idx = torch.tensor(
        [rows[i] for i in range(N)], device=device, dtype=torch.int64
    )
    vals_codec_order = values.gather(1, perm)  # reorder to the emission order
    dense_b = scatter2d(dec_idx, vals_codec_order, C)

    assert torch.allclose(dense_a, dense_b, atol=1e-6), "dense scatter mismatch"


@pytest.mark.parametrize("device", device_params())
def test_cpu_reference_decoder_equivalence(device):
    """
    The CPU reference encoder should decode to the same per-row indices
    as the new encode_batch_rows (not necessarily byte-identical payload).
    """
    N, C, K = 6, 128, 10  # C=128 is divisible by both 64 and 128
    K = make_even_k(K)
    # Generate unique indices per row
    idx = torch.zeros((N, K), device=device, dtype=torch.int64)
    for i in range(N):
        idx[i] = torch.randperm(C, device=device, dtype=torch.int64)[:K]

    # new path
    payload_new, perm_new, _ = encode_batch_rows(idx, C=C)
    rows_new, Cn, Nn = decode_batch_rows(payload_new)
    assert Cn == C and Nn == N
    # ref path
    payload_ref, _meta_ref = encode_batch_rows_cpu(
        idx.detach().cpu().numpy().astype(np.int64), C=C
    )
    rows_ref, Cr, Nr = decode_batch_rows(payload_ref)
    assert Cr == C and Nr == N

    # compare decoded rows (order must be the same since both encoders emit the same ordering)
    for i in range(N):
        assert rows_ref[i] == rows_new[i], f"row {i} decode differs (CPU ref vs new)"
    # permutations must reorder original to decoded
    for i in range(N):
        orig = idx[i].detach().cpu().tolist()
        perm_i = perm_new[i].detach().cpu().tolist()
        reindexed = [orig[p] for p in perm_i]
        assert reindexed == rows_new[i]


# -------------------------------------------------------------------------
# Edge cases & error handling
# -------------------------------------------------------------------------


@pytest.mark.parametrize("device", device_params())
def test_zero_rows(device):
    C, K = 64, 6  # C=64 is divisible by 64
    K = make_even_k(K)
    idx = torch.empty(0, K, dtype=torch.int64, device=device)

    payload, perm, meta = encode_batch_rows(idx, C=C)
    rows, C2, N2 = decode_batch_rows(payload)
    assert C2 == C and N2 == 0
    assert perm.shape == idx.shape
    assert rows == []
    assert "B_hist" in meta and sum(meta["B_hist"].values()) == 0


@pytest.mark.parametrize("device", device_params())
def test_zero_k(device):
    """
    k == 0 should still produce a valid payload and 0-length rows;
    permutation is [N, 0].
    """
    N, C, K = 3, 128, 0  # C=128 is divisible by both 64 and 128
    idx = torch.empty(N, K, dtype=torch.int64, device=device)

    payload, perm, _ = encode_batch_rows(idx, C=C)
    rows, C2, N2 = decode_batch_rows(payload)
    assert C2 == C and N2 == N
    assert perm.shape == (N, 0)
    for i in range(N):
        assert rows[i] == []


@pytest.mark.parametrize("device", device_params())
def test_non_int64_indices_cast_ok(device):
    """
    encode_batch_rows should accept integer tensors not strictly int64
    and cast internally without error.
    """
    N, C, K = 4, 128, 6  # C=128 is divisible by both 64 and 128
    K = make_even_k(K)
    # Generate unique indices per row
    idx_64 = torch.zeros((N, K), device=device, dtype=torch.int64)
    for i in range(N):
        idx_64[i] = torch.randperm(C, device=device, dtype=torch.int64)[:K]
    idx = idx_64.to(torch.int32)

    payload, perm, _ = encode_batch_rows(idx, C=C)
    rows, C2, N2 = decode_batch_rows(payload)
    assert C2 == C and N2 == N
    for i in range(N):
        assert len(rows[i]) == K


def test_invalid_b_choices_raise_for_new_encoder():
    """
    New encoder returns ValueError when no valid B in B_choices.
    (CPU reference falls back to power-of-two divisors, tested below.)
    """
    N, C, K = 2, 10, 4  # C=10 is not divisible by 64 or 128
    # Generate unique indices per row
    idx = torch.zeros((N, K), dtype=torch.int64)
    for i in range(N):
        idx[i] = torch.randperm(C, dtype=torch.int64)[:K]
    with pytest.raises(ValueError, match="No valid B choices for C"):
        encode_batch_rows(
            idx, C=C, B_choices=(3, 6, 12)
        )  # none is a power-of-two divisor of 10


def test_cpu_reference_fallback_works_with_invalid_b_choices():
    """
    CPU reference should still work (it falls back to power-of-two divisors).
    """
    N, C, K = 2, 10, 4  # C=10 is not divisible by 64 or 128
    rows_np = np.random.randint(0, C, size=(N, K), dtype=np.int64)
    payload, meta = encode_batch_rows_cpu(rows_np, C=C, B_choices=(3, 6, 12))
    rows, C2, N2 = decode_batch_rows(payload)
    assert C2 == C and N2 == N
    assert "B_hist" in meta and sum(meta["B_hist"].values()) == N


# -------------------------------------------------------------------------
# Bitmap vs local payload path selection
# -------------------------------------------------------------------------


def test_uses_bitmap_when_dense_within_subbucket():
    """
    Construct a case where k is large within one B=64 sub-bucket,
    so bitmap (B bits) is cheaper than emitting locs (k * lb).
    We verify 'use_bitmap' bit in the row header.
    """
    N, C, B = 1, 128, 64
    # put many positions inside sub 0 of B=64 (enough to make bitmap worthwhile)
    idx = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int64
    )
    payload, perm, _ = encode_batch_rows(idx, C=C, B_choices=(B,))
    C2, N2, row_len, lb, k_param, use_bitmap = parse_first_row_header(payload)
    assert (
        C2 == C and N2 == 1 and lb == int(math.ceil(math.log2(B)))
    )  # lb should be 6 for B=64
    assert use_bitmap == 1, "Expected bitmap path for dense sub-bucket"


def test_uses_local_when_sparse_within_subbucket():
    """
    Construct a case where very few locs within a B=64 block
    makes local payload (k * lb) cheaper than bitmap (B bits).
    """
    N, C, B = 1, 128, 64
    idx = torch.tensor([[0, 63]], dtype=torch.int64)  # very sparse within the block
    payload, perm, _ = encode_batch_rows(idx, C=C, B_choices=(B,))
    C2, N2, row_len, lb, k_param, use_bitmap = parse_first_row_header(payload)
    assert (
        C2 == C and N2 == 1 and lb == int(math.ceil(math.log2(B)))
    )  # lb should be 6 for B=64
    assert use_bitmap == 0, "Expected local (loc-stream) path for sparse sub-bucket"


# -------------------------------------------------------------------------
# Cross-device parity (optional, only when CUDA is available)
# -------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_vs_cpu_decode_equivalence():
    """
    If CUDA is available, the CPU and CUDA encodes should decode equivalently.
    """
    torch.manual_seed(0)
    N, C, K = 7, 128, 10  # C=128 is divisible by both 64 and 128
    K = make_even_k(K)

    # Generate unique indices per row
    idx_cpu = torch.zeros((N, K), device="cpu", dtype=torch.int64)
    for i in range(N):
        idx_cpu[i] = torch.randperm(C, device="cpu", dtype=torch.int64)[:K]
    idx_gpu = idx_cpu.to("cuda")

    payload_cpu, perm_cpu, _ = encode_batch_rows(idx_cpu, C=C)
    payload_gpu, perm_gpu, _ = encode_batch_rows(idx_gpu, C=C)

    rows_cpu, Cc, Nc = decode_batch_rows(payload_cpu)
    rows_gpu, Cg, Ng = decode_batch_rows(payload_gpu)

    assert (Cc, Nc) == (C, N) and (Cg, Ng) == (C, N)

    # decoded rows must match exactly
    assert rows_cpu == rows_gpu

    # permutations must reorder original to decoded in both cases
    for i in range(N):
        orig = idx_cpu[i].tolist()
        re_cpu = [orig[p] for p in perm_cpu[i].cpu().tolist()]
        re_gpu = [orig[p] for p in perm_gpu[i].cpu().tolist()]
        assert re_cpu == rows_cpu[i]
        assert re_gpu == rows_cpu[i]
