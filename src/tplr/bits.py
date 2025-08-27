import math
from numpy._typing._array_like import NDArray
import random
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Literal

import numpy as np
from joblib import Parallel, delayed

# ---------- Bit I/O ----------


def _get_bit_mask(n: int) -> int:
    """Creates a bitmask of n bits."""
    if n < 64:
        return (1 << n) - 1
    # For n >= 64, `1 << n` can be problematic in some Python/C contexts.
    # This creates the same mask value `(2**n) - 1` without causing an overflow.
    mask = 1 << (n - 1)
    return mask + (mask - 1)


class BitWriter:
    """placeholder"""

    def __init__(self):
        self.buf: bytearray = bytearray()
        self.cur: int = 0
        self.nbits: int = 0  # bits in cur

    def write_bits(self, value: int, n: int) -> None:
        """placeholder"""
        if n <= 0:
            return

        mask = _get_bit_mask(n)

        self.cur |= (int(value) & mask) << self.nbits
        self.nbits += n
        while self.nbits >= 8:
            self.buf.append(self.cur & 0xFF)
            self.cur >>= 8
            self.nbits -= 8

    def write_unary(self, q: int) -> None:
        """placeholder"""
        # q ones then a zero
        while q >= 64:
            self.write_bits(0xFFFFFFFFFFFFFFFF, 64)
            q -= 64
        if q > 0:
            self.write_bits((1 << q) - 1, q)
        self.write_bits(0, 1)

    def bits_written(self) -> int:
        """placeholder"""
        return len(self.buf) * 8 + self.nbits

    def flush(self) -> bytes:
        """placeholder"""
        if self.nbits > 0:
            self.buf.append(self.cur & 0xFF)
            self.cur = 0
            self.nbits = 0
        return bytes(self.buf)

    def append_chunk(self, chunk_buf: bytearray, chunk_nbits: int, chunk_cur: int):
        """Appends a chunk of bits from another BitWriter's state."""
        if self.nbits == 0:
            self.buf.extend(chunk_buf)
            self.nbits = chunk_nbits
            self.cur = chunk_cur
        else:
            for byte in chunk_buf:
                self.write_bits(byte, 8)
            if chunk_nbits > 0:
                self.write_bits(chunk_cur, chunk_nbits)


class BitReader:
    """placeholder"""

    def __init__(self, data: bytes):
        self.data: bytes = data
        self.i: int = 0
        self.cur: int = 0
        self.nbits: int = 0

    def _fill(self, need: int) -> None:
        """placeholder"""
        while self.nbits < need and self.i < len(self.data):
            self.cur |= self.data[self.i] << self.nbits
            self.nbits += 8
            self.i += 1

    def read_bits(self, n: int) -> int:
        """placeholder"""
        if n <= 0:
            return 0
        self._fill(n)
        if self.nbits < n:
            raise EOFError("Not enough bits")

        mask = _get_bit_mask(n)

        v = self.cur & mask
        self.cur >>= n
        self.nbits -= n
        return v

    def read_unary(self) -> int:
        """placeholder"""
        q = 0
        while True:
            self._fill(64)
            if self.nbits == 0:
                raise EOFError("EOF while reading unary")

            # Create a mask for the number of available bits
            if self.nbits < 64:
                mask = (1 << self.nbits) - 1
            else:
                mask = 0xFFFFFFFFFFFFFFFF

            # Look for a zero only within the available bits
            masked_cur = self.cur & mask
            
            # Invert bits to find the first 0 (which becomes a 1)
            inverted = ~masked_cur & mask

            if inverted == 0:
                # No zero in the available bits, consume all and loop
                q += self.nbits
                self.nbits = 0
                self.cur = 0
                continue
            
            # Find position of the least significant bit that is 1
            lsb_pos = (inverted & -inverted).bit_length() - 1

            # We found the terminating zero
            q += lsb_pos
            self.nbits -= lsb_pos + 1
            self.cur >>= lsb_pos + 1
            return q


# ---------- Rice coding ----------


def rice_k_from_mean(lmbda: float) -> int:
    """placeholder"""
    if lmbda <= 0.0:
        return 0
    return max(0, round(math.log2(max(lmbda, 1e-9))))


def rice_write(bw: BitWriter, x: int, k: int) -> None:
    """placeholder"""
    if x < 0:
        raise ValueError("Rice expects non-negative")
    m = 1 << k
    q = x // m
    r = x & (m - 1)
    bw.write_unary(q)
    bw.write_bits(r, k)


def rice_read(br: BitReader, k: int) -> int:
    """placeholder"""
    q = br.read_unary()
    if k == 0:
        return q
    r = br.read_bits(k)
    return (q << k) | r


def rice_bits(x: int, k: int) -> int:
    """Returns number of bits for Rice-coded x with param k."""
    if x < 0:
        raise ValueError("Rice expects non-negative")
    m = 1 << k
    q = x // m
    return q + 1 + k


# ---------- Per-row encoder with subchunks (count-first; omit mode if empty) ----------


def instantiate_subs(
    B: int, C: int, idx_sorted: "np.ndarray[Any, Any]"
) -> list["np.ndarray[Any, Any]"]:
    """placeholder"""
    n_sub = C // B
    subs: list["np.ndarray[Any, Any]"] = [
        np.array([], dtype=np.int64) for _ in range(n_sub)
    ]
    # Using groupby on a NumPy array is not ideal, but it's a pragmatic choice
    # to avoid a more complex rewrite of this specific part.
    for j, group in groupby(idx_sorted, key=lambda x: x // B):
        subs[j] = np.array([v % B for v in group], dtype=np.int64)
    return subs


def check_values(B: int, C: int, indices: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
    """placeholder"""
    if C % B != 0 or (B & (B - 1)) != 0:
        raise ValueError("B must be power-of-two dividing C")

    if indices.size > 0 and (np.min(indices) < 0 or np.max(indices) >= C):
        raise ValueError("Index out of range")

    return indices


def _encode_row_into(
    indices: "np.ndarray[Any, Any]",
    C: int,
    B: int,
    use_dense_bitmap: bool,
    bitmap_threshold: int | None = None,
    k_override: int | None = None,
) -> "BitWriter":
    """
    Writes one row into `bw` and returns (bits_added, meta_row).
    """
    lb = B.bit_length() - 1
    # The input array is assumed to be sorted by the caller.
    idx_sorted = indices
    subs = instantiate_subs(B, C, idx_sorted)

    if k_override is not None:
        k = k_override
    else:
        lmbda = (len(indices) / C) * B
        k = rice_k_from_mean(lmbda)

    if bitmap_threshold is None:
        bitmap_threshold = max(1, (B // max(1, lb)))

    bw = BitWriter()
    # Write header for this row
    bw.write_bits(lb, 5)
    bw.write_bits(k, 4)
    bw.write_bits(1 if use_dense_bitmap else 0, 1)

    use_bitmap = [
        (use_dense_bitmap and len(sub_n) >= bitmap_threshold) for sub_n in subs
    ]

    for sub_n, use_bmap in zip(subs, use_bitmap):
        s_j = len(sub_n)
        rice_write(bw, s_j, k)
        if s_j == 0:
            continue

        if use_bmap:
            # Using np.sum with uint64 can lead to overflow if B > 64.
            # By casting to object, we force numpy to use Python's arbitrary-precision
            # integers for the calculation, preventing overflow and ensuring correctness.
            bitmask = np.sum(1 << sub_n.astype(object))
            bw.write_bits(int(bitmask), B)
        else:
            for loc in sub_n:
                bw.write_bits(loc.item(), lb)

    return bw


def _best_row_variant(
    indices: "np.ndarray[Any, Any]",
    C: int,
    B_choices: tuple[int, ...],
    use_dense_bitmap: bool,
    bitmap_threshold: int | None,
) -> tuple[int, dict[str, Any], "BitWriter"]:
    """
    Try multiple B and pick the shortest (in bits).

    Returns:
        (best_B, meta_row, bits).
    """
    best: tuple[float, dict[str, Any], BitWriter | None] = (math.inf, {}, None)

    for b_ in B_choices:
        # This temporary writer is only for measurement.
        bw = _encode_row_into(
            indices,
            C=C,
            B=b_,
            use_dense_bitmap=use_dense_bitmap,
            bitmap_threshold=bitmap_threshold,
        )
        bits = bw.bits_written()

        if bits < best[0]:
            best = (bits, {"B": b_, "use_dense_bitmap": use_dense_bitmap}, bw)

    if best[0] == math.inf:
        if indices.size == 0:
            b_ = B_choices[0]
            bw = _encode_row_into(
                np.array([], dtype=np.int64),
                C=C,
                B=b_,
                use_dense_bitmap=use_dense_bitmap,
                bitmap_threshold=bitmap_threshold,
            )
            best = (bw.bits_written(), {"B": b_, "use_dense_bitmap": use_dense_bitmap}, bw)
        else:
            raise ValueError("Could not find a best encoding variant for a non-empty row.")

    assert best[2] is not None, "BitWriter should not be None at this point"
    return best[1]["B"], best[1], best[2]


# ---------- Batch encoder/decoder ----------


def _derive_bitmap_threshold(B: int, lb: int) -> int:
    """placeholder"""
    # default bitmap threshold = arg where bitmap bits <= locals bits (ignoring count)
    return max(1, (B // max(lb, 1)))


@dataclass
class EncodeMeta:
    """Metadata for batch encoding."""

    C: int
    N: int
    scheme: Literal["per_row", "global"]
    # Global scheme fields
    B: int | None = None
    k: int | None = None
    # Per-row scheme fields
    B_choices: tuple[int, ...] | None = None
    # Meta-mode dependent fields
    total_bits: int | None = None
    avg_bits_per_row: float | None = None
    row_bits: list[int] | None = None
    B_hist: dict[int, int] | None = None
    row_b_codes: bytes | None = None
    rows: list[dict[str, int]] | None = None


def _encode_batch_per_row(
    bw: BitWriter,
    rows: "np.ndarray[Any, Any]",
    C: int,
    B_choices: tuple[int, ...],
    use_dense_bitmap: bool,
    bitmap_threshold: int | None,
    meta_mode: str,
    padding_val: int = -1,
) -> EncodeMeta:
    """
    Processes a NumPy array input for the 'per_row' scheme.
    This version writes directly to the main bitstream, avoiding chunking.
    """
    N = rows.shape[0]
    row_list = [r[r != padding_val] for r in rows]

    if N == 0:
        # Create and return metadata for an empty batch
        meta = EncodeMeta(C=C, N=0, scheme="per_row", B_choices=B_choices)
        meta.total_bits = 0
        meta.avg_bits_per_row = 0.0
        if meta_mode == "summary":
            meta.B_hist = {}
        elif meta_mode == "compact":
            meta.row_b_codes = b""
            meta.row_bits = []
        elif meta_mode == "full":
            meta.rows = []
        return meta

    # Determine the best encoding parameters for each row without writing yet.
    results = [
        _best_row_variant(r, C, B_choices, use_dense_bitmap, bitmap_threshold)
        for r in row_list
    ]

    # Unpack the results: (best_B, meta, bw)
    _, per_row_meta, row_bws = zip(*results)
    row_bits = [b.bits_written() for b in row_bws]

    # Now, iterate again and write each row to the main bit writer using its best parameters.
    for row_bw in row_bws:
        bw.append_chunk(row_bw.buf, row_bw.nbits, row_bw.cur)

    # --- Metadata Generation ---
    b_hist: dict[int, int] = {}
    for meta_item in per_row_meta:
        b_hist[meta_item["B"]] = b_hist.get(meta_item["B"], 0) + 1

    b_choice_indices: list[int] = []
    if B_choices:
        for m in per_row_meta:
            try:
                b_choice_indices.append(B_choices.index(m["B"]))
            except ValueError:
                b_choice_indices.append(0)

    total_payload_bits = bw.bits_written()
    avg_bits_per_row = (np.sum(row_bits) / max(1, N)) if N else 0.0

    meta = EncodeMeta(C=C, N=N, scheme="per_row", B_choices=B_choices)
    if meta_mode == "summary":
        meta.B_hist = b_hist
        meta.total_bits = total_payload_bits
        meta.avg_bits_per_row = avg_bits_per_row
    elif meta_mode == "compact":
        meta.row_b_codes = bytes(b_choice_indices)
        meta.row_bits = row_bits
    elif meta_mode == "full":
        meta.rows = per_row_meta

    return meta


def _get_s_counts_for_B(B, C, N, valid_indices, row_indices_flat, bitmap_threshold):
    """Helper for parallel execution."""
    lb = B.bit_length() - 1
    n_sub = C // B
    derived_bitmap_threshold = (
        bitmap_threshold
        if bitmap_threshold is not None
        else _derive_bitmap_threshold(B, lb)
    )
    sub_indices_flat = valid_indices // B
    s_counts = np.histogram2d(
        row_indices_flat,
        sub_indices_flat,
        bins=(N, n_sub),
        range=[[-0.5, N - 0.5], [-0.5, n_sub - 0.5]],
    )[0].astype(np.int64)
    return {
        "B": B,
        "s_counts": s_counts,
        "lb": lb,
        "n_sub": n_sub,
        "derived_bitmap_threshold": derived_bitmap_threshold,
    }


def _find_best_k_for_B_meta(B_meta, candidate_ks):
    """Helper for parallel execution."""
    best_k_for_B = -1
    best_bits_for_B = float("inf")
    for k in candidate_ks:
        total_bits_per_row = _calculate_batch_bits_np(
            B_meta["s_counts"],
            B_meta["B"],
            k,
            B_meta["derived_bitmap_threshold"],
        )
        current_total_bits = total_bits_per_row.sum()
        if current_total_bits < best_bits_for_B:
            best_bits_for_B = current_total_bits
            best_k_for_B = k
    return (best_bits_for_B, B_meta["B"], best_k_for_B)


def _encode_batch_global(
    bw: BitWriter,
    rows: "np.ndarray[Any, Any]",
    C: int,
    B_choices: tuple[int, ...],
    bitmap_threshold: int | None,
    B_fixed: int | None,
    k_fixed: int | None,
    meta_mode: str,
    heuristic_sample_size: int | None,
    padding_val: int = -1,
) -> "EncodeMeta":
    """NumPy-optimized helper for global scheme."""
    N, _ = rows.shape
    if B_fixed is not None and ((B_fixed & (B_fixed - 1)) != 0 or C % B_fixed != 0):
        raise ValueError("B_fixed must be a power-of-two dividing C")

    mask = rows != padding_val
    row_indices_flat, _ = np.where(mask)
    valid_indices = rows[mask]

    candidate_Bs = [B_fixed] if B_fixed is not None else list(B_choices)
    candidate_ks = [k_fixed] if k_fixed is not None else list(range(0, 9))

    B_metas = [
        _get_s_counts_for_B(
            B, C, N, valid_indices, row_indices_flat, bitmap_threshold
        )
        for B in candidate_Bs
    ]
    results = [
        _find_best_k_for_B_meta(B_meta, candidate_ks) for B_meta in B_metas
    ]

    # Phase 3: Reduce to find best overall
    if not results:
        raise ValueError("Could not determine best B and k")
    _, best_B, best_k = min(results, key=lambda x: x[0])

    # --- Actual encoding with best_B and best_k ---
    lb = best_B.bit_length() - 1
    bw.write_bits(lb, 5)
    bw.write_bits(best_k, 4)

    derived_bitmap_threshold = (
        bitmap_threshold
        if bitmap_threshold is not None
        else _derive_bitmap_threshold(best_B, lb)
    )

    # Pre-calculate all sub-chunk data for all rows to avoid repeated work in the loop.
    all_subs = []
    for i in range(N):
        row_data = rows[i, rows[i] != padding_val]
        all_subs.append(instantiate_subs(best_B, C, row_data))

    row_bits_list = []
    for i in range(N):
        subs = all_subs[i]
        use_bitmap = [len(sub_n) >= derived_bitmap_threshold for sub_n in subs]
        start_bits = bw.bits_written()
        for sub_n, use_bmap in zip(subs, use_bitmap):
            s_j = len(sub_n)
            rice_write(bw, s_j, best_k)
            if s_j == 0:
                continue
            if use_bmap:
                # Using np.sum with uint64 can lead to overflow if B > 64.
                # By casting to object, we force numpy to use Python's arbitrary-precision
                # integers for the calculation, preventing overflow and ensuring correctness.
                bitmask = np.sum(1 << sub_n.astype(object))
                bw.write_bits(int(bitmask), best_B)
            else:
                for loc in sub_n:
                    bw.write_bits(loc.item(), lb)
        row_bits_list.append(bw.bits_written() - start_bits)

    total_payload_bits = bw.bits_written()
    avg_bits_per_row = float(np.mean(row_bits_list)) if row_bits_list else 0.0

    meta = EncodeMeta(C=C, N=N, scheme="global", B=best_B, k=best_k)
    if meta_mode in ("summary", "compact", "full"):
        meta.total_bits = total_payload_bits
        meta.avg_bits_per_row = avg_bits_per_row
        meta.row_bits = row_bits_list
    return meta


def encode_batch(
    rows: "np.ndarray[Any, Any]",
    C: int = 4096,
    B_choices: tuple[int, ...] = (32, 64, 128),
    use_dense_bitmap: bool = True,
    bitmap_threshold: int | None = None,
    *,
    scheme: Literal["per_row", "global"] = "per_row",
    B_fixed: int | None = None,
    k_fixed: int | None = None,
    meta_mode: Literal["none", "summary", "compact", "full"] = "summary",
    heuristic_sample_size: int | None = None,
) -> tuple[bytes, EncodeMeta]:
    """
    Encode a batch of rows (each row is an iterable of indices in [0,C)).
    Global header:
      Cminus1(12), N(16), scheme(1)
      If scheme=1 ("global"): lb(5), k(4)
    scheme:
      - "per_row" (default, scheme bit=0): Each row writes its own lb and k header and,
        for non-empty subchunks, a mode bit.
      - "global" (scheme bit=1): One global lb and k are written once; rows do not write
        per-row lb/k or per-subchunk mode bits.
    meta_mode:
      - "none": minimal meta {C,N,scheme[,B,k]}
      - "summary": lightweight aggregate stats; no per-row list-of-dicts
      - "compact": per-row data stored densely (bytes for B indices, list[int] for row bits)
      - "full": detailed per-row meta (backward-compatible with earlier version)
    Returns: (payload, meta)
    """
    if not isinstance(rows, np.ndarray) or rows.ndim != 2:
        raise ValueError("Input `rows` must be a 2D NumPy array.")

    if scheme not in ("per_row", "global"):
        raise ValueError("scheme must be 'per_row' or 'global'")

    N = rows.shape[0]
    if N >= (1 << 16):
        raise ValueError("N too large (max 65535)")

    rows = np.sort(rows, axis=1)

    bw = BitWriter()
    # Global header (C and N)
    bw.write_bits(C - 1, 12)
    bw.write_bits(N, 16)

    if scheme == "global":
        bw.write_bits(1, 1)
        meta = _encode_batch_global(
            bw,
            rows,
            C,
            B_choices,
            bitmap_threshold,
            B_fixed,
            k_fixed,
            meta_mode,
            heuristic_sample_size,
        )
    else:  # per_row
        bw.write_bits(0, 1)
        meta = _encode_batch_per_row(
            bw,
            rows,
            C,
            B_choices,
            use_dense_bitmap,
            bitmap_threshold,
            meta_mode,
        )

    return bw.flush(), meta


def _calculate_batch_bits_np(
    s_counts: "np.ndarray[Any, Any]", B: int, k: int, bitmap_threshold: int
) -> "np.ndarray[Any, Any]":
    """Vectorized calculation of bit costs for a batch of rows."""
    lb = B.bit_length() - 1

    # Vectorize rice_bits
    m = 1 << k
    q = s_counts // m
    rice_costs = q + 1 + k

    # Calculate payload costs
    use_bitmap = s_counts >= bitmap_threshold
    local_costs = s_counts * lb
    bitmap_costs = np.full_like(s_counts, B, dtype=np.int64)
    payload_costs = np.where(use_bitmap, bitmap_costs, local_costs)
    payload_costs[s_counts == 0] = 0  # No payload for empty subchunks

    return rice_costs + payload_costs


def _decode_batch_per_row(
    br: BitReader, N: int, C: int
) -> list[list[int]]:
    """Helper for per-row scheme, reading from a continuous stream."""
    out: list[list[int]] = []
    for _ in range(N):
        # Each row's data is read directly from the main BitReader
        lb = br.read_bits(5)
        k = br.read_bits(4)
        use_dense_bitmap = br.read_bits(1) == 1
        B = 1 << lb
        if B <= 0 or C % B != 0:
            raise ValueError("Invalid (B,C) in row header")
        n_sub = C // B

        row = []
        bitmap_threshold = _derive_bitmap_threshold(B, lb)
        for j in range(n_sub):
            s_j = rice_read(br, k)
            if s_j == 0:
                continue

            use_bmap = use_dense_bitmap and (s_j >= bitmap_threshold)
            if use_bmap:
                bitmask = br.read_bits(B)
                while bitmask:
                    lsb = bitmask & -bitmask
                    pos = lsb.bit_length() - 1
                    row.append(j * B + pos)
                    bitmask ^= lsb
            else:
                for _k in range(s_j):
                    loc = br.read_bits(lb)
                    row.append(j * B + loc)
        out.append(row)
    return out


def _decode_batch_global(br: BitReader, N: int, C: int) -> list[list[int]]:
    """Helper for global scheme."""
    lb = br.read_bits(5)
    k = br.read_bits(4)
    B = 1 << lb
    if B <= 0 or C % B != 0:
        raise ValueError("Invalid (B,C) in global header")
    n_sub = C // B
    bitmap_threshold = _derive_bitmap_threshold(B, lb)

    out: list[list[int]] = []
    for _ in range(N):
        row: list[int] = []
        # Since we don't know the chunk length, we must decode row by row from the main stream
        for j in range(n_sub):
            s_j = rice_read(br, k)
            if s_j == 0:
                continue
            use_bitmap = s_j >= bitmap_threshold
            if use_bitmap:
                bitmask = br.read_bits(B)
                while bitmask:
                    lsb = bitmask & -bitmask
                    pos = lsb.bit_length() - 1
                    row.append(j * B + pos)
                    bitmask ^= lsb
            else:
                for _k in range(s_j):
                    loc = br.read_bits(lb)
                    row.append(j * B + loc)
        out.append(row)
    return out


def decode_batch(payload: bytes, max_len: int | None = None) -> "np.ndarray[np.int64, Any]":
    """
    Decodes a payload into a 2D NumPy array, padding rows with -1.

    Args:
        payload (bytes): The encoded data.
        max_len (int, optional): The maximum row length for padding. If not provided,
                                 it will be inferred from the longest decoded row,
                                 which can be inefficient.

    Returns:
        np.ndarray: The decoded rows as a 2D NumPy array.
    """
    br = BitReader(payload)
    C = br.read_bits(12) + 1
    N = br.read_bits(16)

    scheme = br.read_bits(1)
    if scheme == 1:
        # This path still returns list[list[int]], needs conversion
        decoded_rows_list = _decode_batch_global(br, N, C)
    else:
        # This path also returns list[list[int]]
        decoded_rows_list = _decode_batch_per_row(br, N, C)

    if max_len is None:
        max_len = max(len(r) for r in decoded_rows_list) if decoded_rows_list else 0

    # Create a padded NumPy array
    np_array = np.full((N, max_len), -1, dtype=np.int64)
    for i, row in enumerate(decoded_rows_list):
        np_array[i, : len(row)] = row

    return np_array


# ---------- Quick demo ----------


def gen_batch(
    N=5000, C=4096, s=32, clustered=False, seed=0
) -> "np.ndarray[Any, Any]":
    """placeholder"""
    rng = random.Random(seed)
    rows = []
    if clustered:
        for _ in range(N):
            centers = [rng.randrange(0, C) for _ in range(max(1, s // 16))]
            pts = set()
            while len(pts) < s:
                c = rng.choice(centers)
                v = max(0, min(C - 1, c + rng.randrange(-16, 17)))
                pts.add(v)
            rows.append(sorted(pts))
    else:
        for _ in range(N):
            rows.append(sorted(rng.sample(range(C), s)))

    max_len = max(len(r) for r in rows) if rows else 0
    padded_rows = np.full((N, max_len), -1, dtype=np.int64)
    for i, row in enumerate(rows):
        padded_rows[i, : len(row)] = row
    return padded_rows
