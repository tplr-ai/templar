# The MIT License (MIT)
# © 2025 tplr.ai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch


def pack_12bit_indices(indices: torch.Tensor) -> torch.Tensor:
    """
    Legacy helper: Pack int64 indices into a 12‑bit representation (pairs → 3 bytes).
    Requires an even count of indices and values < 4096.
    """
    max_idx = indices.max().item() if indices.numel() > 0 else 0
    if max_idx >= 4096:
        raise ValueError(f"Index {max_idx} exceeds 12-bit limit (4095)")

    flat = indices.flatten()
    n = flat.numel()
    if n % 2 != 0:
        raise ValueError(f"Number of indices must be even, got {n}")

    flat = flat.to(torch.int32)
    n_pairs = n // 2
    packed = torch.zeros(n_pairs * 3, dtype=torch.uint8, device=indices.device)

    if n_pairs > 0:
        pairs = flat.reshape(-1, 2)
        idx1 = pairs[:, 0]
        idx2 = pairs[:, 1]
        packed[0::3] = (idx1 & 0xFF).to(torch.uint8)
        packed[1::3] = (((idx1 >> 8) & 0x0F) | ((idx2 & 0x0F) << 4)).to(torch.uint8)
        packed[2::3] = ((idx2 >> 4) & 0xFF).to(torch.uint8)

    return packed


def unpack_12bit_indices(
    packed: torch.Tensor, values_shape: tuple[int, ...]
) -> torch.Tensor:
    """
    Legacy helper: Unpack 12‑bit representation back into int64 indices and reshape
    to the provided `values_shape` (which must match the original indices shape).
    """
    device = packed.device
    n_indices = 1
    for d in values_shape:
        n_indices *= int(d)
    if n_indices == 0:
        return torch.zeros(values_shape, dtype=torch.int64, device=device)
    if n_indices % 2 != 0:
        raise ValueError(f"Number of indices must be even, got {n_indices}")

    out = torch.zeros(n_indices, dtype=torch.int64, device=device)
    n_pairs = n_indices // 2
    if n_pairs > 0:
        b0 = packed[0::3].to(torch.int64)
        b1 = packed[1::3].to(torch.int64)
        b2 = packed[2::3].to(torch.int64)

        out[0::2] = b0 | ((b1 & 0x0F) << 8)
        out[1::2] = ((b1 >> 4) & 0x0F) | (b2 << 4)
    return out.view(*values_shape)
