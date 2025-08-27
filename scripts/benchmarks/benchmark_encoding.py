import time
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tplr.bits import gen_batch, encode_batch, decode_batch

async def benchmark_encoding(scheme, N, C, s, executor, semaphore):
    print(f"--- Benchmarking {scheme} scheme (N={N}, C={C}, s={s}) ---")
    
    # Generate data
    original_batch = gen_batch(N=N, C=C, s=s)

    original_batch = original_batch - original_batch[:,:1]
    original_batch = original_batch[:,1:]

    kwargs = {}
    if scheme == "global":
        kwargs["heuristic_sample_size"] = 2048
    
    # Warm-up run
    await encode_batch(original_batch, C=C, scheme=scheme, executor=executor, semaphore=semaphore, **kwargs)
    
    # Timed run
    payload = None
    meta = None
    start_time = time.time()
    for _ in range(5):
        payload, meta = await encode_batch(original_batch, C=C, scheme=scheme, executor=executor, semaphore=semaphore, **kwargs)
    end_time = time.time()
    
    duration = end_time - start_time
    avg_time = duration / 5
    
    print(f"Average encoding time: {avg_time:.4f} seconds")

    # Decode and verify the last payload
    if payload:
        decoded_batch_padded = decode_batch(payload, max_len=original_batch.shape[1])
        
        # Trim padding from the decoded batch to match the original's structure
        # This is necessary because the decoder always returns a rectangular array
        decoded_rows = []
        for row in decoded_batch_padded:
            # Find the first occurrence of -1 (padding) and trim the row
            non_padded_part = row[row != -1]
            decoded_rows.append(non_padded_part)

        # Reconstruct the decoded batch to match the original's ragged nature for comparison
        # We create a new padded array based on the actual lengths of decoded rows
        max_decoded_len = max(len(r) for r in decoded_rows) if decoded_rows else 0
        decoded_batch_for_comparison = np.full((len(decoded_rows), max_decoded_len), -1, dtype=np.int64)
        for i, row in enumerate(decoded_rows):
            decoded_batch_for_comparison[i, :len(row)] = row

        # Ensure the reconstructed decoded batch has the same shape as the original
        if decoded_batch_for_comparison.shape[1] < original_batch.shape[1]:
            pad_width = original_batch.shape[1] - decoded_batch_for_comparison.shape[1]
            decoded_batch_for_comparison = np.pad(
                decoded_batch_for_comparison, 
                ((0,0), (0, pad_width)), 
                'constant', 
                constant_values=-1
            )

        assert np.array_equal(original_batch, decoded_batch_for_comparison), "Roundtrip failed: Decoded data does not match original"
        print("Roundtrip validation PASSED")
    else:
        print("Skipping roundtrip validation (no payload generated)")

    print("-" * 40)

async def main():
    C = 4096
    
    # User requested benchmark
    N = 51840
    s = 128

    executor = ThreadPoolExecutor()
    semaphore = asyncio.Semaphore(20)

    await benchmark_encoding("per_row", N, C, s, executor, semaphore)
    await benchmark_encoding("global", N, C, s, executor, semaphore)

if __name__ == "__main__":
    asyncio.run(main())
