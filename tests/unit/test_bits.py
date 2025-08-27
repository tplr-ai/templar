import numpy as np
import pytest

# Set numpy print options to be verbose for debugging
np.set_printoptions(threshold=np.inf)

from tplr.bits import decode_batch, encode_batch, gen_batch


@pytest.mark.parametrize("scheme", ["global", "per_row"])
@pytest.mark.parametrize("N", [10, 100])
@pytest.mark.parametrize("s", [16, 32])
@pytest.mark.parametrize("C", [4096])
@pytest.mark.parametrize("clustered", [False, True])
def test_numpy_roundtrip(scheme, N, s, C, clustered):
    """
    Tests that encoding a NumPy array and decoding it back yields the original data.
    """
    # 1. Generate original data
    original_rows_np = gen_batch(N=N, C=C, s=s, clustered=clustered, seed=0)

    # 2. Encode the NumPy array
    payload, meta = encode_batch(
        original_rows_np,
        C=C,
        scheme=scheme,
        meta_mode="full",
    )

    # 3. Decode the payload back to a NumPy array
    decoded_rows_np = decode_batch(payload, max_len=original_rows_np.shape[1])

    # 4. Verify that the decoded data matches the original data
    np.testing.assert_array_equal(
        original_rows_np,
        decoded_rows_np,
        err_msg="Decoded NumPy array does not match the original.",
    )
