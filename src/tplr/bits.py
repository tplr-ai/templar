import math
import random
from dataclasses import dataclass
from itertools import groupby
from typing import Literal

import numpy as np
from joblib import Parallel, delayed

# ---------- Bit I/O ----------


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
        self.cur |= (value & ((1 << n) - 1)) << self.nbits
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
        mask = (1 << n) - 1
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

            # Invert bits to find the first 0 (which becomes a 1)
            inverted = ~self.cur
            # Find position of the least significant bit that is 1
            lsb_pos = (inverted & -inverted).bit_length() - 1

            if lsb_pos < self.nbits:
                # We found the terminating zero within the available bits
                q += lsb_pos
                self.nbits -= lsb_pos + 1
                self.cur >>= lsb_pos + 1
                return q
            else:
                # No zero in the current buffer, so all are ones. Consume all.
                q += self.nbits
                self.nbits = 0
                self.cur = 0
                # Loop to _fill more data


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


def _encode_row_into(
    bw: BitWriter,
    indices: list[int],
    C: int,
    B: int,
    use_dense_bitmap: bool = True,
    bitmap_threshold: int | None = None,
    k_override: int | None = None,
) -> tuple[int, dict[str, int]]:
    """
    Writes one row into `bw` and returns (bits_added, meta_row).
    Format per row:
      lb(5), k(4),
      for j in 0..(C/B-1):
        s_j (Rice(k))
        if s_j>0:
           mode_bit (1=bitmap, 0=locals)
           if mode=0: s_j * lb bits (local indices)
           if mode=1: B bits bitmap
    """
    lb = math.ceil(math.log2(B))
    idx_sorted = check_and_sort_values(B, C, indices)
    subs = instantiate_subs(B, C, idx_sorted)

    # choose Rice k from per-subchunk mean unless overridden
    lmbda = (len(idx_sorted) / C) * B
    k = rice_k_from_mean(lmbda) if k_override is None else k_override

    # default bitmap threshold = arg where bitmap bits <= locals bits (ignoring count)
    if bitmap_threshold is None:
        bitmap_threshold = max(1, (B // lb))

    start_bits = bw.bits_written()
    # Small row header
    bw.write_bits(lb, 5)
    bw.write_bits(k, 4)

    use_bitmap = use_dense_bitmap and (len(subs) >= bitmap_threshold)
    bw.write_bits(1 if use_bitmap else 0, 1)
    use_bitmap = [use_bitmap for _ in range(len(subs))]

    # Payload
    bw = write_bytes_loop(
        bw,
        k,
        subs,
        B,
        lb,
        use_bitmap,
    )

    bits_added = bw.bits_written() - start_bits
    return bits_added, {"B": B, "lb": lb, "k": k, "bitmap_threshold": bitmap_threshold}


def _calculate_row_bits_from_subs(
    subs: list[list[int]],
    B: int,
    lb: int,
    k: int,
    bitmap_threshold: int,
) -> int:
    """placeholder"""
    total_bits = 0
    use_bitmap = [len(sub_n) >= bitmap_threshold for sub_n in subs]
    for sub_n, use_bmap in zip(subs, use_bitmap):
        s_j = len(sub_n)
        total_bits += rice_bits(s_j, k)
        if s_j == 0:
            continue
        total_bits += B if use_bmap else s_j * lb
    return total_bits


def _encode_row_global_into(
    bw: BitWriter,
    indices: list[int],
    C: int,
    B: int,
    k: int,
    bitmap_threshold: int | None = None,
) -> int:
    """
    Encode a row with a fixed global B and k.
    Differences from _encode_row_into:
      - No per-row header (lb,k) written here
      - No per-subchunk mode bit; decoder derives mode deterministically from threshold
    Returns bits_added.
    """
    lb = math.ceil(math.log2(B))
    idx_sorted = check_and_sort_values(B, C, indices)
    subs = instantiate_subs(B, C, idx_sorted)

    if bitmap_threshold is None:
        bitmap_threshold = _derive_bitmap_threshold(B, lb)

    start_bits = bw.bits_written()

    use_bitmap = [len(sub_n) >= bitmap_threshold for sub_n in subs]

    # Payload
    bw = write_bytes_loop(
        bw,
        k,
        subs,
        B,
        lb,
        use_bitmap,
    )

    return bw.bits_written() - start_bits


def write_bytes_loop(
    bw: BitWriter,
    k: int,
    subs: list[list[int]],
    B: int,
    lb: int,
    use_bitmap: list[bool],
) -> BitWriter:
    """placeholder"""
    for sub_n, use_bmap in zip(subs, use_bitmap):
        s_j = len(sub_n)
        rice_write(bw, s_j, k)
        if s_j == 0:
            continue  # omit mode bit for empties

        if use_bmap:
            bitmask = 0
            for loc in sub_n:
                bitmask |= 1 << loc
            bw.write_bits(bitmask, B)
        else:
            for loc in sub_n:
                bw.write_bits(loc, lb)
    return bw


def instantiate_subs(B: int, C: int, idx_sorted: list[int]) -> list[list[int]]:
    """placeholder"""
    n_sub = C // B
    subs: list[list[int]] = [[] for _ in range(n_sub)]
    for j, group in groupby(idx_sorted, key=lambda x: x // B):
        subs[j].extend([v % B for v in group])
    return subs


def check_and_sort_values(B: int, C: int, indices: list[int]) -> list[int]:
    """placeholder"""
    if C % B != 0 or (B & (B - 1)) != 0:
        raise ValueError("B must be power-of-two dividing C")

    _ = indices.sort()
    if min(indices) < 0 or max(indices) >= C:
        raise ValueError("Index out of range")

    return indices


def _best_row_variant(
    indices: list[int], C: int, B_choices: tuple[int, ...]
) -> tuple[int, dict[str, int], tuple[int, int, int]]:
    """
    Try multiple B and pick the shortest (in bits).

        Returns
            (best_B, meta_row, (lb,k,bitmap_threshold)).
    """
    # Dry-run encodes into throwaway writers to measure bits; then caller re-encodes for real.
    best: tuple[float, dict[str, int]] = (math.inf, {})
    for b_ in B_choices:
        tmp = BitWriter()
        bits, meta = _encode_row_into(tmp, indices, C=C, B=b_)
        if bits < best[0]:
            best = (bits, meta)
    if best[0] == math.inf:
        raise ValueError("Best not found")
    meta = best[1]
    return meta["B"], meta, (meta["lb"], meta["k"], meta["bitmap_threshold"])


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


# def encode_rowwise_inner_loop(
#     row: list[int],
#     C: int,
#     B_choices: tuple[int, ...],
#     use_dense_bitmap,
#     bitmap_threshold,
# ):
#     best_B, _, _ = _best_row_variant(r, C=C, B_choices=B_choices)
#     return _encode_row_into(
#         bw,
#         row,
#         C=C,
#         B=best_B,
#         use_dense_bitmap=use_dense_bitmap,
#         bitmap_threshold=bitmap_threshold,
#     )


def _encode_batch_per_row(
    bw: BitWriter,
    row_list: list[list[int]],
    C: int,
    B_choices: tuple[int, ...],
    use_dense_bitmap: bool,
    bitmap_threshold: int | None,
    meta_mode: str,
) -> EncodeMeta:
    """Helper for per-row scheme."""
    N = len(row_list)
    per_row_meta: list[dict[str, int]] = []
    b_hist: dict[int, int] = {}
    b_choice_indices: list[int] = []
    row_bits: list[int] = []

    for r in row_list:
        best_B, _, _ = _best_row_variant(r, C=C, B_choices=B_choices)
        bits_added, meta_row = _encode_row_into(
            bw,
            r,
            C=C,
            B=best_B,
            use_dense_bitmap=use_dense_bitmap,
            bitmap_threshold=bitmap_threshold,
        )
        per_row_meta.append({"bits": bits_added, **meta_row})
        row_bits.append(bits_added)
        b_hist[best_B] = b_hist.get(best_B, 0) + 1
        try:
            b_choice_indices.append(B_choices.index(best_B))
        except ValueError:
            b_choice_indices.append(0)

    payload = bw.flush()
    total_payload_bits = len(payload) * 8
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


def _encode_batch_global(
    bw: BitWriter,
    row_list: list[list[int]],
    C: int,
    B_choices: tuple[int, ...],
    bitmap_threshold: int | None,
    B_fixed: int | None,
    k_fixed: int | None,
    meta_mode: str,
    heuristic_sample_size: int | None,
) -> EncodeMeta:
    """Helper for global scheme."""
    N = len(row_list)
    if B_fixed is not None and ((B_fixed & (B_fixed - 1)) != 0 or C % B_fixed != 0):
        raise ValueError("B_fixed must be a power-of-two dividing C")

    # If a heuristic sample size is given, use a random subset of rows for the search
    search_list = row_list
    if (
        heuristic_sample_size is not None
        and B_fixed is None
        and heuristic_sample_size < N
    ):
        search_list = random.sample(row_list, heuristic_sample_size)

    candidate_Bs = [B_fixed] if B_fixed is not None else list(B_choices)
    candidate_ks = [k_fixed] if k_fixed is not None else list(range(0, 9))

    best_total_bits = None
    best_B = None
    best_k = None

    for B in candidate_Bs:
        if B is None:
            continue
        lb = math.ceil(math.log2(B))
        if bitmap_threshold is None:
            derived_bitmap_threshold = _derive_bitmap_threshold(B, lb)
        else:
            derived_bitmap_threshold = bitmap_threshold

        with Parallel(n_jobs=20, prefer="threads") as parallel:
            # Pre-calculate subs for all rows for this B
            subs_by_row = parallel(
                delayed(instantiate_subs)(B, C, check_and_sort_values(B, C, r))
                for r in search_list
            )

            for k in candidate_ks:
                if k is None:
                    continue

                row_bits: list[int] = parallel(  # type: ignore
                    delayed(_calculate_row_bits_from_subs)(
                        subs, B, lb, k, derived_bitmap_threshold
                    )
                    for subs in subs_by_row
                )
                tmp_total = np.sum(row_bits)

                if (best_total_bits is None) or (tmp_total < best_total_bits):
                    best_total_bits = tmp_total
                    best_B = B
                    best_k = k

    if best_B is None or best_k is None:
        raise ValueError(f"Best of k or B is None: {best_B=} {best_k=}")

    lb = math.ceil(math.log2(best_B))
    bw.write_bits(lb, 5)
    bw.write_bits(best_k, 4)

    row_bits: list[int] = []
    for r in row_list:
        bits_added = _encode_row_global_into(
            bw, r, C=C, B=best_B, k=best_k, bitmap_threshold=bitmap_threshold
        )
        row_bits.append(bits_added)

    payload = bw.flush()
    total_payload_bits = len(payload) * 8
    avg_bits_per_row = (np.sum(row_bits) / max(1, N)) if N else 0.0

    meta = EncodeMeta(
        C=C,
        N=N,
        scheme="global",
        B=best_B,
        k=best_k,
    )
    if meta_mode in ("summary", "compact", "full"):
        meta.total_bits = total_payload_bits
        meta.avg_bits_per_row = avg_bits_per_row
        meta.row_bits = row_bits
    return meta


def encode_batch(
    rows: list[list[int]],
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

    if scheme not in ("per_row", "global"):
        raise ValueError("scheme must be 'per_row' or 'global'")

    # Normalize rows to list of lists for single pass
    row_list: list[list[int]] = [list(map(int, r)) for r in rows]
    N = len(row_list)
    if N >= (1 << 16):
        raise ValueError("N too large (max 65535)")

    bw = BitWriter()
    # Global header (C and N)
    bw.write_bits(C - 1, 12)
    bw.write_bits(N, 16)

    if scheme == "global":
        bw.write_bits(1, 1)
        meta = _encode_batch_global(
            bw,
            row_list,
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
            row_list,
            C,
            B_choices,
            use_dense_bitmap,
            bitmap_threshold,
            meta_mode,
        )

    return bw.flush(), meta


def _decode_batch_per_row(br: BitReader, N: int, C: int) -> list[list[int]]:
    """Helper for per-row scheme."""
    out: list[list[int]] = []
    for _ in range(N):
        lb = br.read_bits(5)
        k = br.read_bits(4)
        B = 1 << lb
        if B <= 0 or C % B != 0:
            raise ValueError("Invalid (B,C) in row header")
        n_sub = C // B

        row = []
        mode = br.read_bits(1)
        for j in range(n_sub):
            s_j = rice_read(br, k)
            if s_j == 0:
                continue

            if mode == 1:
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


def decode_batch(payload: bytes) -> list[list[int]]:
    """placeholder"""
    br = BitReader(payload)
    C = br.read_bits(12) + 1
    N = br.read_bits(16)

    # scheme flag: 0=per_row, 1=global
    scheme = br.read_bits(1)
    if scheme == 1:
        return _decode_batch_global(br, N, C)
    return _decode_batch_per_row(br, N, C)


# ---------- Quick demo ----------


def gen_batch(N=5000, C=4096, s=32, clustered=False, seed=0):
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
    return rows
