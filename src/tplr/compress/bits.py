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

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

# -------------------------- Bit I/O --------------------------------------


class BitWriter:
    def __init__(self) -> None:
        self.buf: bytearray = bytearray()
        self.cur: int = 0
        self.nbits: int = 0

    def write_bits(self, value: int, n: int) -> None:
        if n <= 0:
            return
        self.cur |= (int(value) & ((1 << n) - 1)) << self.nbits
        self.nbits += n
        while self.nbits >= 8:
            self.buf.append(self.cur & 0xFF)
            self.cur >>= 8
            self.nbits -= 8

    def write_unary(self, q: int) -> None:
        # q ones then a zero
        while q >= 64:
            self.write_bits(0xFFFFFFFFFFFFFFFF, 64)
            q -= 64
        if q > 0:
            self.write_bits((1 << q) - 1, q)
        self.write_bits(0, 1)

    def flush(self) -> bytes:
        if self.nbits > 0:
            self.buf.append(self.cur & 0xFF)
            self.cur = 0
            self.nbits = 0
        return bytes(self.buf)


class BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.idx = 0
        self.cur = 0
        self.nbits = 0

    def _fill(self, n: int) -> None:
        while self.nbits < n and self.idx < len(self.data):
            self.cur |= int(self.data[self.idx]) << self.nbits
            self.idx += 1
            self.nbits += 8

    def read_bits(self, n: int) -> int:
        if n <= 0:
            return 0
        self._fill(n)
        mask = (1 << n) - 1
        out = self.cur & mask
        self.cur >>= n
        self.nbits -= n
        return out

    def read_unary(self) -> int:
        q = 0
        while True:
            bit = self.read_bits(1)
            if bit == 0:
                break
            q += 1
        return q

    def read_bytes(self, n: int) -> bytes:
        out = bytearray()
        for _ in range(n):
            out.append(self.read_bits(8))
        return bytes(out)


# -------------------------- Rice helpers ---------------------------------


def _rice_k_from_mean(lmbda: float) -> int:
    if lmbda <= 0.0:
        return 0
    return max(0, round(math.log2(max(lmbda, 1e-9))))


def _rice_write(bw: BitWriter, x: int, k: int) -> None:
    m = 1 << k
    q = x // m
    r = x & (m - 1)
    bw.write_unary(q)
    bw.write_bits(r, k)


def _rice_read(br: BitReader, k: int) -> int:
    q = br.read_unary()
    r = br.read_bits(k) if k > 0 else 0
    return (q << k) + r


# -------------------------------------------------------------------------
# CPU reference encoder (kept for tests/tools)
# -------------------------------------------------------------------------


def encode_batch_rows_cpu(
    rows_np: np.ndarray,
    *,
    C: int,
    B_choices: tuple[int, ...] = (64, 128),
    scheme: str = "per_row",
    workers: int | None = None,
) -> tuple[bytes, dict]:
    if scheme != "per_row":
        raise ValueError("Only scheme='per_row' is implemented")
    valid_B: list[int] = [
        b for b in B_choices if b > 0 and (b & (b - 1)) == 0 and (C % b) == 0
    ]
    if not valid_B:
        b = 1
        valid_B = []
        while b <= C:
            if C % b == 0 and (b & (b - 1)) == 0:
                valid_B.append(b)
            b <<= 1

    def encode_one_row(row_indices: np.ndarray) -> tuple[bytes, int, int]:
        krow = int(row_indices.size)
        best_bits = None
        best_B = None
        best_info = None
        for B in valid_B:
            lb = int(math.ceil(math.log2(B)))
            n_sub = C // B
            js = (row_indices // B).astype(np.int64)
            counts = np.bincount(js, minlength=n_sub)
            lmbda = (krow / max(1, C)) * B
            k_param = _rice_k_from_mean(lmbda)
            header = 5 + 4 + 1
            rb_sum = 0
            for c in counts.tolist():
                m = 1 << k_param
                q = int(c) // m
                rb_sum += q + 1 + k_param
            s_nonzero = int((counts > 0).sum())
            bits_local = header + rb_sum + int(lb * int(counts.sum()))
            bits_bitmap = header + rb_sum + int(B * s_nonzero)
            cur_bits = min(bits_local, bits_bitmap)
            if best_bits is None or cur_bits < best_bits:
                best_bits = cur_bits
                best_B = B
                best_info = {
                    "lb": lb,
                    "k": k_param,
                    "use_bitmap": (bits_bitmap < bits_local),
                    "B": B,
                }

        assert best_info is not None and best_B is not None

        row_bw = BitWriter()
        lb = best_info["lb"]
        k_param = best_info["k"]
        use_bitmap = best_info["use_bitmap"]
        B = best_info["B"]
        n_sub = C // B
        js = (row_indices // B).astype(np.int64)
        locs = (row_indices - js * B).astype(np.int64)
        order = np.argsort(js)
        js_sorted = js[order]
        locs_sorted = locs[order]
        sub_lists: list[np.ndarray] = [None] * n_sub  # type: ignore[assignment]
        for j in range(n_sub):
            s = int(np.searchsorted(js_sorted, j, side="left"))
            e = int(np.searchsorted(js_sorted, j, side="right"))
            if e > s:
                sub_lists[j] = np.sort(locs_sorted[s:e])
            else:
                sub_lists[j] = np.empty((0,), dtype=np.int64)

        row_bw.write_bits(lb, 5)
        row_bw.write_bits(k_param, 4)
        row_bw.write_bits(1 if use_bitmap else 0, 1)
        for j in range(n_sub):
            sl = sub_lists[j]
            s_len = int(sl.size)
            _rice_write(row_bw, s_len, k_param)
            if s_len == 0:
                continue
            if use_bitmap:
                bitmask = 0
                for loc in sl.tolist():
                    bitmask |= 1 << int(loc)
                row_bw.write_bits(bitmask, B)
            else:
                for loc in sl.tolist():
                    row_bw.write_bits(int(loc), lb)

        return row_bw.flush(), best_bits if best_bits is not None else 0, best_B  # type: ignore[return-value]

    N = rows_np.shape[0]
    bw = BitWriter()
    bw.write_bits(C - 1, 12)
    bw.write_bits(N, 16)
    bw.write_bits(0, 1)

    row_bits: list[int] = []
    B_hist: dict[int, int] = {}
    max_workers = workers if workers and workers > 0 else min(32, os.cpu_count() or 8)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for row_bytes, bits_used, B_used in ex.map(
            encode_one_row, (rows_np[i] for i in range(N))
        ):
            bw.write_bits(len(row_bytes), 16)
            for byte in row_bytes:
                bw.write_bits(int(byte), 8)
            row_bits.append(bits_used)
            B_hist[B_used] = B_hist.get(B_used, 0) + 1

    payload = bw.flush()
    meta = {
        "total_bits": len(payload) * 8,
        "avg_bits_per_row": (sum(row_bits) / max(1, N)) if N else 0.0,
        "B_hist": B_hist,
    }
    return payload, meta


@torch.no_grad()
def encode_batch_rows(
    idx: torch.Tensor,  # [rows, k] int64 on CPU or CUDA
    *,
    C: int,
    B_choices: tuple[int, ...] = (64, 128),
) -> tuple[bytes, dict]:
    """
    Rice/bitmap encoder.

    Returns:
      payload: bytes
      meta:    dict with basic stats
    """
    # Normalize dtype & capture device
    if idx.dtype != torch.int64:
        idx = idx.to(torch.int64)
    rows, k = idx.shape
    device = idx.device

    # --- pick best B per row (vectorised on GPU) ------------------------
    B_sorted = tuple(
        sorted([b for b in B_choices if b > 0 and (C % b) == 0 and (b & (b - 1)) == 0])
    )
    if len(B_sorted) == 0:
        raise ValueError("No valid B choices for C")

    header = 5 + 4 + 1
    best_bits = torch.full((rows,), 1 << 60, device=device, dtype=torch.int64)
    best_B = torch.full((rows,), B_sorted[0], device=device, dtype=torch.int64)
    best_use_bitmap = torch.zeros((rows,), device=device, dtype=torch.bool)

    Bmin = B_sorted[0]
    if all((b % Bmin) == 0 for b in B_sorted):
        n_sub_min = C // Bmin
        js_min = (idx // Bmin).to(torch.int64)  # [rows, k]
        counts_min = (
            F.one_hot(js_min, num_classes=n_sub_min).sum(dim=1).to(torch.int64)
        )  # [rows, n_sub_min]
        lmbda_base = k / max(1, C)

        for B in B_sorted:
            g = B // Bmin
            if g == 1:
                counts_B = counts_min
            else:
                counts_B = counts_min.reshape(rows, n_sub_min // g, g).sum(
                    dim=2
                )  # [rows, n_sub]

            lb = int(math.ceil(math.log2(B)))
            n_sub = C // B
            k_param = int(max(0, round(math.log2(max(lmbda_base * B, 1e-9)))))
            m = 1 << k_param
            q = counts_B // m
            rb_sum = q.sum(dim=1) + (1 + k_param) * n_sub  # [rows]
            nonzero = (counts_B > 0).sum(dim=1)  # [rows]
            bits_local = header + rb_sum + lb * k
            bits_bitmap = header + rb_sum + B * nonzero
            cur_bits = torch.minimum(bits_local, bits_bitmap).to(torch.int64)
            use_bitmap = bits_bitmap < bits_local
            update = cur_bits < best_bits
            best_bits = torch.where(update, cur_bits, best_bits)
            best_B = torch.where(update, torch.full_like(best_B, B), best_B)
            best_use_bitmap = torch.where(update, use_bitmap, best_use_bitmap)
    else:
        # fallback: evaluate each B independently
        for B in B_sorted:
            lb = int(math.ceil(math.log2(B)))
            n_sub = C // B
            js = (idx // B).to(torch.int64)
            row_ids = torch.arange(rows, device=device, dtype=torch.int64).unsqueeze(1)
            flat = (row_ids * n_sub + js).reshape(-1)
            counts = torch.bincount(flat, minlength=rows * n_sub).reshape(rows, n_sub)
            lmbda = (k / max(1, C)) * B
            k_param = int(max(0, round(math.log2(max(lmbda, 1e-9)))))
            m = 1 << k_param
            q = counts // m
            rb_sum = q.sum(dim=1) + (1 + k_param) * n_sub
            nonzero = (counts > 0).sum(dim=1)
            bits_local = header + rb_sum + lb * k
            bits_bitmap = header + rb_sum + B * nonzero
            cur_bits = torch.minimum(bits_local, bits_bitmap).to(torch.int64)
            use_bitmap = bits_bitmap < bits_local
            update = cur_bits < best_bits
            best_bits = torch.where(update, cur_bits, best_bits)
            best_B = torch.where(update, torch.full_like(best_B, B), best_B)
            best_use_bitmap = torch.where(update, use_bitmap, best_use_bitmap)

    # --- produce payload ------------------------------------------------
    bw = BitWriter()
    bw.write_bits(C - 1, 12)
    bw.write_bits(rows, 16)
    bw.write_bits(0, 1)  # reserved

    for B in B_sorted:
        row_mask = best_B == B
        if not row_mask.any():
            continue
        idx_sub = idx[row_mask]  # [R_b, k]
        R_b = idx_sub.shape[0]
        lb = int(math.ceil(math.log2(B)))
        n_sub = C // B
        lmbda = (k / max(1, C)) * B
        k_param = int(max(0, round(math.log2(max(lmbda, 1e-9)))))
        use_bitmap_rows = best_use_bitmap[row_mask]  # [R_b]

        j = idx_sub // B  # [R_b, k]
        loc = idx_sub - j * B  # [R_b, k]
        # Group by sub-chunk id; sort by j then sort loc within each sub-chunk.
        order = torch.argsort(j, dim=1, stable=True)  # [R_b, k]
        j_sorted_cpu = torch.gather(j, 1, order).detach().cpu()
        loc_sorted_cpu = torch.gather(loc, 1, order).detach().cpu()

        for r in range(R_b):
            row_bw = BitWriter()
            row_bw.write_bits(lb, 5)
            row_bw.write_bits(k_param, 4)
            use_bitmap = bool(use_bitmap_rows[r].item())
            row_bw.write_bits(1 if use_bitmap else 0, 1)

            js_np = j_sorted_cpu[r].numpy()
            locs_np = loc_sorted_cpu[r].numpy()

            # Count occurrences per sub with numpy bincount (fast)
            counts = np.bincount(js_np, minlength=n_sub)
            # Write rice lengths + payload sub-by-sub
            base = 0
            for sub in range(n_sub):
                s_len = int(counts[sub])
                _rice_write(row_bw, s_len, k_param)
                if s_len == 0:
                    continue
                ran = slice(base, base + s_len)
                base += s_len
                # within each sub, ensure ascending loc order
                sub_locs = locs_np[ran]
                sub_locs_sorted = np.sort(sub_locs, kind="stable")
                if use_bitmap:
                    bitmask = 0
                    for locv in sub_locs_sorted.tolist():
                        bitmask |= 1 << int(locv)
                    row_bw.write_bits(bitmask, B)
                else:
                    for locv in sub_locs_sorted.tolist():
                        row_bw.write_bits(int(locv), lb)

            # commit row chunk
            row_bytes = row_bw.flush()
            bw.write_bits(len(row_bytes), 16)
            for byte in row_bytes:
                bw.write_bits(int(byte), 8)

    payload = bw.flush()
    meta = {
        "total_bits": len(payload) * 8,
        "avg_bits_per_row": float(best_bits.float().mean().item()),
        "B_hist": {int(b): int((best_B == b).sum().item()) for b in B_sorted},
    }
    return payload, meta


# -------------------------------------------------------------------------
# Decoder (CPU)
# -------------------------------------------------------------------------


def decode_batch_rows(payload: bytes) -> tuple[list[list[int]], int, int]:
    """
    Decode payload created by encode_batch_rows(...).
    Returns (rows, C, N) where `rows` is a list of per-row global indices.
    """
    br = BitReader(payload)
    C = br.read_bits(12) + 1
    N = br.read_bits(16)
    _ = br.read_bits(1)  # reserved

    rows: list[list[int]] = []
    for _i in range(N):
        row_len = br.read_bits(16)
        row_bytes = br.read_bytes(row_len)
        rr = BitReader(row_bytes)
        lb = rr.read_bits(5)
        k_param = rr.read_bits(4)
        use_bitmap = rr.read_bits(1)
        B = 1 << lb
        n_sub = C // B

        indices: list[int] = []
        for j in range(n_sub):
            s_len = _rice_read(rr, k_param)
            if s_len == 0:
                continue
            if use_bitmap:
                bitmask = rr.read_bits(B)
                for loc in range(B):
                    if (bitmask >> loc) & 1:
                        indices.append(j * B + loc)
            else:
                for _ in range(s_len):
                    loc = rr.read_bits(lb)
                    indices.append(j * B + loc)
        rows.append(indices)
    return rows, C, N
