import pytest
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tplr.bits import encode_batch, decode_batch, gen_batch

@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("C", [4096])
@pytest.mark.parametrize("B", [16, 32])
@pytest.mark.parametrize("s", [10, 100])
@pytest.mark.parametrize("scheme", ["global", "per_row"])
@pytest.mark.asyncio
async def test_numpy_roundtrip(clustered, C, B, s, scheme):
    """
    Tests that encoding and then decoding a batch of rows
    results in the original batch.
    """
    # Generate a batch of rows
    N = 100  # Number of rows in the batch
    batch = gen_batch(N=N, C=C, s=s, clustered=clustered, seed=0)

    executor = ThreadPoolExecutor()
    semaphore = asyncio.Semaphore(20)

    # Encode the batch
    payload, _ = await encode_batch(batch, C=C, B_choices=(B,), scheme=scheme, executor=executor, semaphore=semaphore)

    # Decode the batch
    decoded_batch = decode_batch(payload, max_len=batch.shape[1])

    # Check that the decoded batch is the same as the original
    assert np.array_equal(batch, decoded_batch), "Decoded batch does not match original"
