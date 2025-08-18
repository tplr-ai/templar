import pytest
from src.tplr.bits import (
    BitWriter,
    BitReader,
    rice_write,
    rice_read,
    encode_batch,
    decode_batch,
    gen_batch,
    rice_k_from_mean,
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


def test_rice_coding():
    for k in range(5):
        for val in [0, 1, 5, 10, 100, 123]:
            bw = BitWriter()
            rice_write(bw, val, k)
            data = bw.flush()
            br = BitReader(data)
            decoded_val = rice_read(br, k)
            assert decoded_val == val


def test_rice_k_from_mean():
    assert rice_k_from_mean(0.5) == 0
    assert rice_k_from_mean(1) == 0
    assert rice_k_from_mean(2) == 1
    assert rice_k_from_mean(3.9) == 2
    assert rice_k_from_mean(4.1) == 2
    assert rice_k_from_mean(8) == 3


@pytest.mark.parametrize("scheme", ["per_row", "global"])
@pytest.mark.parametrize("s", [0, 1, 16, 32])
@pytest.mark.parametrize("clustered", [False, True])
def test_encode_decode_roundtrip(scheme, s, clustered):
    C = 4096
    N = 100
    batch = gen_batch(N=N, C=C, s=s, clustered=clustered, seed=42)
    
    enc, meta = encode_batch(batch, C=C, scheme=scheme, meta_mode="summary")
    dec = decode_batch(enc)
    
    assert batch == dec, f"Failed with scheme={scheme}, s={s}, clustered={clustered}"


def test_encode_batch_empty():
    enc, meta = encode_batch([], C=4096)
    dec = decode_batch(enc)
    assert dec == []
    assert meta["N"] == 0


def test_encode_batch_invalid_inputs():
    with pytest.raises(ValueError):
        encode_batch([[-1]], C=4096)  # Index out of range
    with pytest.raises(ValueError):
        encode_batch([[4096]], C=4096) # Index out of range
    with pytest.raises(ValueError):
        encode_batch([[1]], C=4095) # C not power of 2 is not a requirement, but B must divide C
    with pytest.raises(ValueError):
        encode_batch([[1]], C=4096, B_choices=(33,))
    with pytest.raises(ValueError):
        _ = encode_batch([[1]], C=4096, scheme="invalid_scheme")


def test_decode_batch_invalid_data_per_row():
    # Test case where B does not divide C
    bw = BitWriter()
    bw.write_bits(4096, 12)  # C-1 = 4096 -> C=4097
    bw.write_bits(1, 16)     # N=1
    bw.write_bits(0, 1)      # scheme=per_row
    bw.write_bits(5, 5)      # lb=5 -> B=32. 4097 is not divisible by 32
    bw.write_bits(0, 4)      # k=0
    data = bw.flush()
    with pytest.raises(ValueError, match="Invalid .* in row header"):
        decode_batch(data)


def test_decode_batch_invalid_data_global():
    # Test case where B does not divide C
    bw = BitWriter()
    bw.write_bits(4096, 12)  # C-1 = 4096 -> C=4097
    bw.write_bits(1, 16)     # N=1
    bw.write_bits(1, 1)      # scheme=global
    bw.write_bits(5, 5)      # lb=5 -> B=32. 4097 is not divisible by 32
    bw.write_bits(0, 4)      # k=0
    data = bw.flush()
    with pytest.raises(ValueError, match="Invalid .* in global header"):
        decode_batch(data)


def test_meta_modes():
    batch = gen_batch(N=10, C=4096, s=16, seed=1)
    
    _, meta_none = encode_batch(batch, C=4096, meta_mode="none")
    assert "total_bits" not in meta_none
    assert "rows" not in meta_none

    _, meta_summary = encode_batch(batch, C=4096, meta_mode="summary")
    assert "total_bits" in meta_summary
    assert "avg_bits_per_row" in meta_summary
    assert "B_hist" in meta_summary

    _, meta_compact = encode_batch(batch, C=4096, meta_mode="compact")
    assert "row_b_codes" in meta_compact
    assert "row_bits" in meta_compact

    _, meta_full = encode_batch(batch, C=4096, meta_mode="full")
    assert "rows" in meta_full
    assert len(meta_full["rows"]) == 10


def test_global_scheme_fixed_params():
    batch = gen_batch(N=10, C=4096, s=16, seed=1)
    enc, meta = encode_batch(batch, C=4096, scheme="global", B_fixed=64, k_fixed=3)
    dec = decode_batch(enc)
    assert batch == dec
    assert meta["B"] == 64
    assert meta["k"] == 3
