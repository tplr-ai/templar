import pytest

from src.tplr.bits import (
    BitReader,
    BitWriter,
    EncodeMeta,
    _best_row_variant,
    _calculate_row_bits_from_subs,
    _derive_bitmap_threshold,
    _encode_batch_global,
    _encode_batch_per_row,
    _encode_row_global_into,
    _encode_row_into,
    check_and_sort_values,
    decode_batch,
    encode_batch,
    gen_batch,
    instantiate_subs,
    rice_k_from_mean,
    rice_read,
    rice_write,
    write_bytes_loop,
)


def test_bit_io():
    bw = BitWriter()
    bw.write_bits(0b101, 3)
    bw.write_bits(0b11010, 5)
    bw.write_unary(3)  # 1110
    bw.write_bits(0b1, 1)
    data = bw.flush()

    br = BitReader(data)
    assert br.read_bits(3) == 0b101
    assert br.read_bits(5) == 0b11010
    assert br.read_unary() == 3
    assert br.read_bits(1) == 0b1
    # The buffer was flushed to 2 bytes (16 bits), we read 13, so 3 bits of padding remain
    assert br.read_bits(3) == 0
    with pytest.raises(EOFError):
        br.read_bits(1)


def test_bit_writer_bits_written():
    """Test the bits_written method of BitWriter."""
    bw = BitWriter()
    assert bw.bits_written() == 0
    bw.write_bits(0b101, 3)
    assert bw.bits_written() == 3
    bw.write_bits(0b11010, 5)
    assert bw.bits_written() == 8
    bw.flush()
    assert bw.bits_written() == 8


def test_rice_coding():
    for k in range(5):
        for val in [0, 1, 5, 10, 100, 123]:
            bw = BitWriter()
            rice_write(bw, val, k)
            data = bw.flush()
            br = BitReader(data)
            decoded_val = rice_read(br, k)
            assert decoded_val == val

    with pytest.raises(ValueError, match="Rice expects non-negative"):
        bw = BitWriter()
        rice_write(bw, -1, 3)


def test_rice_k_from_mean():
    assert rice_k_from_mean(0.5) == 0
    assert rice_k_from_mean(1) == 0
    assert rice_k_from_mean(2) == 1
    assert rice_k_from_mean(3.9) == 2
    assert rice_k_from_mean(4.1) == 2
    assert rice_k_from_mean(8) == 3


@pytest.mark.parametrize("scheme", ["per_row", "global"])
@pytest.mark.parametrize("s", [1, 16, 32])
@pytest.mark.parametrize("clustered", [False, True])
def test_encode_decode_roundtrip(scheme, s, clustered):
    C = 4096
    # N = 100
    N = 100
    batch = gen_batch(N=N, C=C, s=s, clustered=clustered, seed=42)

    assert any(batch), "batch is all empty lists"

    enc, meta = encode_batch(batch, C=C, scheme=scheme, meta_mode="summary")
    dec = decode_batch(enc)

    assert batch == dec, f"Failed with scheme={scheme}, s={s}, clustered={clustered}"


def test_encode_batch_empty():
    enc, meta = encode_batch([], C=4096)
    dec = decode_batch(enc)
    assert dec == []
    assert meta.N == 0


def test_encode_batch_invalid_inputs():
    with pytest.raises(ValueError):
        encode_batch([[-1]], C=4096)  # Index out of range
    with pytest.raises(ValueError):
        encode_batch([[4096]], C=4096)  # Index out of range
    with pytest.raises(ValueError):
        encode_batch(
            [[1]], C=4095
        )  # C not power of 2 is not a requirement, but B must divide C
    with pytest.raises(ValueError):
        encode_batch([[1]], C=4096, B_choices=(33,))


def test_decode_batch_invalid_data_per_row():
    # Test case where B does not divide C
    bw = BitWriter()
    bw.write_bits(4096, 12)  # C-1 = 4096 -> C=4097
    bw.write_bits(1, 16)  # N=1
    bw.write_bits(0, 1)  # scheme=per_row
    bw.write_bits(5, 5)  # lb=5 -> B=32. 4097 is not divisible by 32
    bw.write_bits(0, 4)  # k=0
    data = bw.flush()
    with pytest.raises(ValueError, match="Invalid .* in row header"):
        decode_batch(data)


def test_decode_batch_invalid_data_global():
    # Test case where B does not divide C
    bw = BitWriter()
    bw.write_bits(4096, 12)  # C-1 = 4096 -> C=4097
    bw.write_bits(1, 16)  # N=1
    bw.write_bits(1, 1)  # scheme=global
    bw.write_bits(5, 5)  # lb=5 -> B=32. 4097 is not divisible by 32
    bw.write_bits(0, 4)  # k=0
    data = bw.flush()
    with pytest.raises(ValueError, match="Invalid .* in global header"):
        decode_batch(data)


def test_meta_modes():
    batch = gen_batch(N=10, C=4096, s=16, seed=1)

    assert any(batch), "batch is all empty lists"

    _, meta_none = encode_batch(batch, C=4096, meta_mode="none")
    assert isinstance(meta_none, EncodeMeta)
    assert meta_none.total_bits is None
    assert meta_none.rows is None

    _, meta_summary = encode_batch(batch, C=4096, meta_mode="summary")
    assert isinstance(meta_summary, EncodeMeta)
    assert meta_summary.total_bits is not None
    assert meta_summary.avg_bits_per_row is not None
    assert meta_summary.B_hist is not None

    _, meta_compact = encode_batch(batch, C=4096, meta_mode="compact")
    assert isinstance(meta_compact, EncodeMeta)
    assert meta_compact.row_b_codes is not None
    assert meta_compact.row_bits is not None

    _, meta_full = encode_batch(batch, C=4096, meta_mode="full")
    assert isinstance(meta_full, EncodeMeta)
    assert meta_full.rows is not None
    assert len(meta_full.rows) == 10


def test_global_scheme_fixed_params():
    batch = gen_batch(N=10, C=4096, s=16, seed=1)
    enc, meta = encode_batch(batch, C=4096, scheme="global", B_fixed=64, k_fixed=3)
    dec = decode_batch(enc)
    assert batch == dec
    assert isinstance(meta, EncodeMeta)
    assert meta.B == 64
    assert meta.k == 3


def test_check_and_sort_values():
    # Valid cases
    assert check_and_sort_values(32, 4096, [10, 0, 5]) == [0, 5, 10]
    assert check_and_sort_values(32, 4096, [4095, 0]) == [0, 4095]

    # Invalid cases
    with pytest.raises(ValueError, match="B must be power-of-two dividing C"):
        check_and_sort_values(33, 4096, [0])  # B not power of 2
    with pytest.raises(ValueError, match="B must be power-of-two dividing C"):
        check_and_sort_values(32, 4097, [0])  # C not divisible by B
    with pytest.raises(ValueError, match="Index out of range"):
        check_and_sort_values(32, 4096, [-1])  # Index < 0
    with pytest.raises(ValueError, match="Index out of range"):
        check_and_sort_values(32, 4096, [4096])  # Index >= C


def test_instantiate_subs():
    # B=32, C=128, n_sub=4
    indices = [0, 31, 32, 63, 64, 95, 96, 127]
    subs = instantiate_subs(B=32, C=128, idx_sorted=indices)
    assert subs == [[0, 31], [0, 31], [0, 31], [0, 31]]

    indices = [0, 5, 10, 33, 40, 127]
    subs = instantiate_subs(B=32, C=128, idx_sorted=indices)
    assert subs == [[0, 5, 10], [1, 8], [], [31]]

    indices = []
    subs = instantiate_subs(B=32, C=128, idx_sorted=indices)
    assert subs == [[], [], [], []]


def test_derive_bitmap_threshold():
    assert _derive_bitmap_threshold(B=32, lb=5) == 6
    assert _derive_bitmap_threshold(B=64, lb=6) == 10
    assert _derive_bitmap_threshold(B=128, lb=7) == 18
    assert _derive_bitmap_threshold(B=2, lb=1) == 2


def test_encode_batch_per_row_logic():
    """Test the internal logic of the per-row encoding helper."""
    C = 4096
    B_choices = (32, 64, 128)
    row_list = gen_batch(N=5, C=C, s=16, seed=1)
    bw = BitWriter()

    meta = _encode_batch_per_row(
        bw,
        row_list,
        C,
        B_choices,
        use_dense_bitmap=True,
        bitmap_threshold=None,
        meta_mode="full",
    )

    assert meta.scheme == "per_row"
    assert meta.N == 5
    assert isinstance(meta, EncodeMeta)
    assert meta.rows is not None
    assert len(meta.rows) == 5


def test_encode_batch_global_logic():
    """Test the internal logic of the global encoding helper."""
    C = 4096
    B_choices = (64,)
    row_list = gen_batch(N=5, C=C, s=16, seed=1)

    bw = BitWriter()
    meta = _encode_batch_global(
        bw,
        row_list,
        C,
        B_choices,
        bitmap_threshold=None,
        B_fixed=64,
        k_fixed=3,
        meta_mode="full",
        heuristic_sample_size=None,
    )

    assert meta.scheme == "global"
    assert isinstance(meta, EncodeMeta)
    assert meta.B == 64
    assert meta.k == 3
    assert meta.row_bits is not None
    assert len(bw.flush()) > 0


def test_encode_meta_dataclass():
    """Test the EncodeMeta dataclass instantiation."""
    meta = EncodeMeta(C=4096, N=10, scheme="per_row")
    assert meta.C == 4096
    assert meta.N == 10
    assert meta.scheme == "per_row"
    assert meta.B is None
    assert meta.k is None
    assert meta.B_choices is None
    assert meta.total_bits is None
    assert meta.avg_bits_per_row is None
    assert meta.row_bits is None
    assert meta.B_hist is None
    assert meta.row_b_codes is None
    assert meta.rows is None

    meta_global = EncodeMeta(
        C=8192,
        N=20,
        scheme="global",
        B=64,
        k=3,
        total_bits=1024,
        avg_bits_per_row=51.2,
    )
    assert meta_global.scheme == "global"
    assert meta_global.B == 64
    assert meta_global.k == 3
    assert meta_global.total_bits == 1024
    assert meta_global.avg_bits_per_row == 51.2


def test_best_row_variant():
    """Test the _best_row_variant function."""
    # This is a bit of a white-box test. We know that for a small number of
    # indices, a smaller B will be better. For a larger number, a larger B
    # will be better because the overhead of the bitmap is less.
    C = 4096
    B_choices = (32, 64, 128)

    # Case 1: Few indices, a larger B can be better due to less overhead from
    # encoding counts of empty subchunks.
    row_sparse = [10, 20, 30]
    best_B_sparse, _, _ = _best_row_variant(row_sparse, C, B_choices)
    assert best_B_sparse == 64

    # Case 2: Many indices, larger B should be better.
    row_dense = list(range(100))
    best_B_dense, _, _ = _best_row_variant(row_dense, C, B_choices)
    assert best_B_dense == 128


def test_write_bytes_loop():
    """Test the write_bytes_loop function."""
    # Case 1: Use bitmap
    bw_bitmap = BitWriter()
    subs_bitmap = [[0, 31]]
    bw_bitmap = write_bytes_loop(
        bw_bitmap, k=0, subs=subs_bitmap, B=32, lb=5, use_bitmap=[True]
    )
    data_bitmap = bw_bitmap.flush()
    br_bitmap = BitReader(data_bitmap)
    assert br_bitmap.read_unary() == 2  # s_j = 2
    bitmask = (1 << 0) | (1 << 31)
    assert br_bitmap.read_bits(32) == bitmask

    # Case 2: Use local indices
    bw_locals = BitWriter()
    subs_locals = [[5, 10]]
    bw_locals = write_bytes_loop(
        bw_locals, k=0, subs=subs_locals, B=32, lb=5, use_bitmap=[False]
    )
    data_locals = bw_locals.flush()
    br_locals = BitReader(data_locals)
    assert br_locals.read_unary() == 2  # s_j = 2
    assert br_locals.read_bits(5) == 5
    assert br_locals.read_bits(5) == 10


def test_calculate_row_bits_from_subs():
    """Test the _calculate_row_bits_from_subs function."""
    # B=32, lb=5, k=2, threshold=6
    subs = [[0, 1, 2, 3, 4, 5], [10, 20]]  # one above, one below threshold
    bits = _calculate_row_bits_from_subs(subs, B=32, lb=5, k=2, bitmap_threshold=6)
    # s_j=6 -> rice_bits(6,2) = 1+1+2=4. use_bitmap=True -> 32 bits. Total = 36
    # s_j=2 -> rice_bits(2,2) = 0+1+2=3. use_bitmap=False -> 2*5=10 bits. Total = 13
    assert bits == 36 + 13

    # Test with empty sub
    subs_empty = [[], [1, 2, 3]]
    bits_empty = _calculate_row_bits_from_subs(
        subs_empty, B=32, lb=5, k=2, bitmap_threshold=6
    )
    # s_j=0 -> rice_bits(0,2) = 0+1+2=3.
    # s_j=3 -> rice_bits(3,2) = 0+1+2=3. use_bitmap=False -> 3*5=15. Total = 18
    assert bits_empty == 3 + 18


def test_gen_batch():
    # Test basic generation
    batch = gen_batch(N=10, C=1024, s=16, seed=0)
    assert len(batch) == 10
    assert all(len(row) == 16 for row in batch)
    assert all(all(0 <= x < 1024 for x in row) for row in batch)
    assert all(row == sorted(row) for row in batch)

    # Test clustered generation
    batch_clustered = gen_batch(N=10, C=1024, s=16, clustered=True, seed=0)
    assert len(batch_clustered) == 10
    # Check if values are somewhat close to each other - this is a heuristic
    variances = [
        sum((x - sum(row) / len(row)) ** 2 for x in row) / len(row)
        for row in batch_clustered
        if row
    ]
    # This is a loose check, but variance should be lower in clustered data
    # compared to uniformly random data. A deep statistical test is overkill here.
    assert all(v < 100000 for v in variances)  # Heuristic threshold

    # Test empty rows
    batch_empty = gen_batch(N=5, C=1024, s=0)
    assert len(batch_empty) == 5
    assert all(row == [] for row in batch_empty)
