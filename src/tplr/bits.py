import math
import random
from collections.abc import Sequence
from typing import Any, Literal

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
        v = value & ((1 << n) - 1)
        self.cur |= v << self.nbits
        self.nbits += n
        while self.nbits >= 8:
            self.buf.append(self.cur & 0xFF)
            self.cur >>= 8
            self.nbits -= 8

    def write_unary(self, q: int) -> None:
        """placeholder"""
        # q ones then a zero
        while q >= 32:
            self.write_bits((1 << 32) - 1, 32)
            q -= 32
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
            self._fill(1)
            if self.nbits == 0:
                raise EOFError("EOF while reading unary")
            bit = self.cur & 1
            self.cur >>= 1
            self.nbits -= 1
            if bit == 1:
                q += 1
            else:
                break
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


# ---------- Per-row encoder with subchunks (count-first; omit mode if empty) ----------


def _encode_row_into(
    bw: BitWriter,
    indices: Sequence[int],
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
        bitmap_threshold = max(1, math.floor(B / lb))

    start_bits = bw.bits_written()
    # Small row header
    bw.write_bits(lb, 5)
    bw.write_bits(k, 4)

    use_bitmap = use_dense_bitmap and (len(subs) >= bitmap_threshold)
    bw.write_bits(1 if use_bitmap else 0, 1)
    use_bitmap = [use_bitmap for _ in range(len(subs))]

    # Payload
    bw = write_bytes_loop(
        bw, k, subs, B, lb, use_bitmap,
    )

    bits_added = bw.bits_written() - start_bits
    return bits_added, {"B": B, "lb": lb, "k": k, "bitmap_threshold": bitmap_threshold}


def _encode_row_global_into(
    bw: BitWriter,
    indices: Sequence[int],
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
        bw, k, subs, B, lb, use_bitmap,
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
    for v in idx_sorted:
        j = v // B
        subs[j].append(v % B)
    return subs


def check_and_sort_values(B: int, C: int, indices: Sequence[int]) -> list[int]:
    """placeholder"""
    if C % B != 0 or (B & (B - 1)) != 0:
        raise ValueError("B must be power-of-two dividing C")

    idx_sorted = sorted(int(v) for v in indices)
    if min(idx_sorted) < 0 or max(idx_sorted) >= C:
        raise ValueError("Index out of range")

    return idx_sorted


def _best_row_variant(
    indices: Sequence[int], C: int, B_choices: tuple[int, ...]
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
    return max(1, math.floor(B / max(lb, 1)))


def encode_batch(
    rows: Sequence[Sequence[int]],
    C: int = 4096,
    B_choices: tuple[int, ...] = (32, 64, 128),
    use_dense_bitmap: bool = True,
    bitmap_threshold: int | None = None,
    *,
    scheme: Literal["per_row", "global"] = "per_row",
    B_fixed: int | None = None,
    k_fixed: int | None = None,
    meta_mode: Literal["none", "summary", "compact", "full"] = "summary",
) -> tuple[bytes, dict[str, Any]]:
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

    if scheme == "per_row":
        # scheme flag 0
        bw.write_bits(0, 1)
        per_row_meta: list[dict[str, int]] = []
        b_hist: dict[int, int] = {}
        b_choice_indices: list[int] = []  # for compact meta
        row_bits: list[int] = []
        for r in row_list:
            # pick best B for this row (dry run)
            best_B, _, _ = _best_row_variant(r, C=C, B_choices=B_choices)
            # encode for real
            bits_added, meta = _encode_row_into(
                bw,
                r,
                C=C,
                B=best_B,
                use_dense_bitmap=use_dense_bitmap,
                bitmap_threshold=bitmap_threshold,
            )
            per_row_meta.append({"bits": bits_added, **meta})
            row_bits.append(bits_added)
            b_hist[best_B] = b_hist.get(best_B, 0) + 1
            # map B to its index in B_choices for compact meta
            try:
                b_choice_indices.append(B_choices.index(best_B))
            except ValueError:
                b_choice_indices.append(0)

        payload = bw.flush()
        total_payload_bits = len(payload) * 8
        avg_bits_per_row = (sum(row_bits) / max(1, N)) if N else 0.0

        # Build meta according to meta_mode
        if meta_mode == "none":
            meta = {"C": C, "N": N, "scheme": "per_row"}
        elif meta_mode == "summary":
            meta = {
                "C": C,
                "N": N,
                "scheme": "per_row",
                "B_choices": B_choices,
                "B_hist": b_hist,
                "total_bits": total_payload_bits,
                "avg_bits_per_row": avg_bits_per_row,
            }
        elif meta_mode == "compact":
            meta = {
                "C": C,
                "N": N,
                "scheme": "per_row",
                "B_choices": B_choices,
                "row_b_codes": bytes(b_choice_indices),
                "row_bits": row_bits,
            }
        elif meta_mode == "full":
            meta = {
                "C": C,
                "N": N,
                "rows": per_row_meta,
                "B_choices": B_choices,
                "scheme": "per_row",
            }

        return payload, meta

    # scheme == "global"
    # Choose global B and k if not fixed
    if B_fixed is not None and ((B_fixed & (B_fixed - 1)) != 0 or C % B_fixed != 0):
        raise ValueError("B_fixed must be a power-of-two dividing C")

    candidate_Bs = [B_fixed] if B_fixed is not None else list(B_choices)
    candidate_ks = [k_fixed] if k_fixed is not None else list(range(0, 9))

    best_total_bits = None
    best_B = None
    best_k = None

    bks = [
        (B, k)
        for B in candidate_Bs
        for k in candidate_ks
        if B is not None and k is not None
    ]
    for B, k in bks:
        tmp_total = 0
        tmp_bw = BitWriter()
        # simulate rows without headers
        for r in row_list:
            tmp_total += _encode_row_global_into(
                tmp_bw, r, C=C, B=B, k=k, bitmap_threshold=bitmap_threshold
            )
        if (best_total_bits is None) or (tmp_total < best_total_bits):
            best_total_bits = tmp_total
            best_B = B
            best_k = k

    assert best_B is not None and best_k is not None

    lb = math.ceil(math.log2(best_B))
    # scheme flag 1
    bw.write_bits(1, 1)
    # global lb and k
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
    avg_bits_per_row = (sum(row_bits) / max(1, N)) if N else 0.0

    if meta_mode == "none":
        meta = {"C": C, "N": N, "scheme": "global", "B": best_B, "k": best_k}
    elif meta_mode == "summary":
        meta = {
            "C": C,
            "N": N,
            "scheme": "global",
            "B": best_B,
            "k": best_k,
            "total_bits": total_payload_bits,
            "avg_bits_per_row": avg_bits_per_row,
        }
    elif meta_mode == "compact":
        meta = {
            "C": C,
            "N": N,
            "scheme": "global",
            "B": best_B,
            "k": best_k,
            "row_bits": row_bits,
        }
    elif meta_mode == "full":
        meta = {
            "C": C,
            "N": N,
            "scheme": "global",
            "B": best_B,
            "k": best_k,
            "row_bits": row_bits,
        }

    return payload, meta


def decode_batch(payload: bytes) -> list[list[int]]:
    """placeholder"""
    br = BitReader(payload)
    C = br.read_bits(12) + 1
    N = br.read_bits(16)
    out: list[list[int]] = []

    # scheme flag: 0=per_row, 1=global
    scheme = br.read_bits(1)
    if scheme == 0:
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
            row.sort()
            out.append(row)
        return out

    # scheme == 1 (global)
    lb = br.read_bits(5)
    k = br.read_bits(4)
    B = 1 << lb
    if B <= 0 or C % B != 0:
        raise ValueError("Invalid (B,C) in global header")
    n_sub = C // B
    bitmap_threshold = _derive_bitmap_threshold(B, lb)

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
        row.sort()
        out.append(row)
    return out


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
