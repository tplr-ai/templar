# TODO: Update r2 with latest fine edu score dataset.
# # ruff: noqa
# import os
# import pytest
# import numpy as np
# import torch
# from pathlib import Path
# from dotenv import load_dotenv

# # Find and load the correct .env file
# env_path = Path(__file__).parent.parent / ".env"
# if not env_path.exists():
#     raise FileNotFoundError(f"Required .env file not found at {env_path}")

# load_dotenv(env_path, override=True)

# # Verify environment variables are loaded
# required_vars = [
#     "R2_DATASET_ACCOUNT_ID",
#     "R2_DATASET_BUCKET_NAME",
#     "R2_DATASET_READ_ACCESS_KEY_ID",
#     "R2_DATASET_READ_SECRET_ACCESS_KEY",
# ]

# missing_vars = [var for var in required_vars if not os.environ.get(var)]
# if missing_vars:
#     raise EnvironmentError(
#         f"Missing required environment variables: {', '.join(missing_vars)}"
#     )

# import torch
# from tplr.logging import logger, debug, T
# from tplr.dataset import DatasetLoader
# from tplr.r2_dataset import R2DatasetLoader
# from tplr.hparams import load_hparams

# # Enable debug logging
# debug()


# def set_random_seeds(seed=42):
#     """Set random seeds for reproducibility"""
#     # GPU optimizations.
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     # random.seed(42)
#     torch.backends.cudnn.deterministic = True


# @pytest.mark.asyncio
# async def test_dataset_equivalence():
#     """
#     Test that DatasetLoader and R2DatasetLoader produce identical outputs
#     given the same input parameters and seed.
#     """
#     start_time = T()
#     logger.info("Starting dataset equivalence test")

#     # Set fixed random seeds
#     set_random_seeds()

#     # Test parameters
#     batch_size = 2
#     sequence_length = 128
#     n_pages = 2
#     seed = 255
#     offset = 0

#     # Load tokenizer
#     hparams = load_hparams()
#     tokenizer = hparams.tokenizer
#     logger.info(f"Tokenizer loaded ({T() - start_time:.2f}s)")

#     # Generate pages using both methods
#     r2_pages = await R2DatasetLoader.next_pages(
#         offset=offset, n_pages=n_pages, seed=seed
#     )
#     set_random_seeds()  # Reset seeds before generating HF pages
#     hf_pages = await DatasetLoader.next_pages(offset=offset, n_pages=n_pages, seed=seed)

#     logger.info(f"R2 pages: {r2_pages}")
#     logger.info(f"HF pages: {hf_pages}")

#     # Assert pages are identical
#     assert r2_pages == hf_pages, (
#         f"Page generation differs between loaders:\nR2: {r2_pages}\nHF: {hf_pages}"
#     )

#     # Create both loaders with identical settings
#     set_random_seeds()  # Reset seeds before creating loaders
#     r2_loader = await R2DatasetLoader.create(
#         batch_size=batch_size,
#         sequence_length=sequence_length,
#         pages_info=r2_pages,
#         tokenizer=tokenizer,
#         pack_samples=False,
#     )

#     set_random_seeds()  # Reset seeds before creating second loader
#     hf_loader = await DatasetLoader.create(
#         batch_size=batch_size,
#         sequence_length=sequence_length,
#         pages_info=hf_pages,
#         tokenizer=tokenizer,
#         pack_samples=False,
#         # shuffle=True,  # Keep shuffle for HF loader if it accepts it
#     )

#     # Convert loaders to lists for deterministic comparison
#     r2_batches = list(r2_loader)
#     hf_batches = list(hf_loader)

#     # Assert same number of batches
#     assert len(r2_batches) == len(hf_batches), (
#         f"Different number of batches: R2={len(r2_batches)}, HF={len(hf_batches)}"
#     )

#     # Compare each batch
#     for batch_idx, (r2_batch, hf_batch) in enumerate(zip(r2_batches, hf_batches)):
#         logger.info(f"Comparing batch {batch_idx}")

#         # Convert to tensors if they aren't already
#         r2_tensor = (
#             torch.tensor(r2_batch)
#             if not isinstance(r2_batch, torch.Tensor)
#             else r2_batch
#         )
#         hf_tensor = (
#             torch.tensor(hf_batch)
#             if not isinstance(hf_batch, torch.Tensor)
#             else hf_batch
#         )

#         # Log shapes
#         logger.info(f"R2 batch shape: {r2_tensor.shape}")
#         logger.info(f"HF batch shape: {hf_tensor.shape}")

#         # Compare batch shapes
#         assert r2_tensor.shape == hf_tensor.shape, (
#             f"Batch {batch_idx} shapes differ: R2={r2_tensor.shape}, HF={hf_tensor.shape}"
#         )

#         # # Compare batch contents
#         # try:
#         #     assert torch.equal(r2_tensor, hf_tensor), \
#         #         f"Batch {batch_idx} contents differ"
#         # except AssertionError:
#         #     # If batches differ, log detailed comparison
#         #     logger.error(f"\nR2 batch:\n{r2_tensor}\n")
#         #     logger.error(f"HF batch:\n{hf_tensor}\n")
#         #     logger.error(f"Difference:\n{(r2_tensor - hf_tensor).abs().sum()}")

#         #     # Compare decoded text for debugging
#         #     for seq_idx in range(min(2, batch_size)):
#         #         r2_text = tokenizer.decode(r2_tensor[seq_idx])
#         #         hf_text = tokenizer.decode(hf_tensor[seq_idx])
#         #         logger.error(f"\nSequence {seq_idx} comparison:")
#         #         logger.error(f"R2: {r2_text[:200]}...")
#         #         logger.error(f"HF: {hf_text[:200]}...")
#         #     raise

#         # # Log success for this batch
#         # logger.info(f"✓ Batch {batch_idx} identical")

#         # # Optional: Compare decoded text for visual verification
#         # if batch_idx == 0:  # Only for first batch
#         #     for seq_idx in range(min(2, batch_size)):
#         #         r2_text = tokenizer.decode(r2_tensor[seq_idx])
#         #         hf_text = tokenizer.decode(hf_tensor[seq_idx])
#         #         logger.info(f"\nSequence {seq_idx} comparison:")
#         #         logger.info(f"R2: {r2_text[:200]}...")
#         #         logger.info(f"HF: {hf_text[:200]}...")

#     logger.info(f"[green]Test completed successfully ({T() - start_time:.2f}s)[/green]")
