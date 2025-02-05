import random
import torch
from tests.mocks.loader import MockLoader

class MockR2DatasetLoader:
    """A mock version of R2DatasetLoader for testing without file I/O."""

    @classmethod
    async def create(cls, batch_size=None, sequence_length=None, pages_info=None, tokenizer=None, pack_samples=True):
        # Return a dummy loader that yields one dummy batch.
        # The dummy batch is a list containing one integer token.
        return MockLoader([[1]])

    @classmethod
    async def next_pages(cls, offset: int, n_pages: int, seed: int):
        # Generate dummy pages as tuples: (config_name, page_number, split).
        pages = [("config1", offset * n_pages + i, "splitA") for i in range(n_pages)]
        return pages

    @classmethod
    async def _load_r2_metadata(cls):
        # Return dummy metadata with expected structure.
        return (
            {
                "config1": {"shards": [{"num_rows": 10, "path": "dummy_path_config1"}]},
                "config2": {"shards": [{"num_rows": 10, "path": "dummy_path_config2"}]},
                "config3": {"shards": [{"num_rows": 10, "path": "dummy_path_config3"}]},
                "config4": {"shards": [{"num_rows": 10, "path": "dummy_path_config4"}]},
            },
            None
        )

    @classmethod
    async def get_loader(cls, window: int, hparams, tokenizer, seed: int = None, data_type: str = "training", pack_samples: bool = True):
        seed_val = seed if seed is not None else random.randint(0, 10000)
        pages = await cls.next_pages(offset=window, n_pages=hparams.pages_per_window, seed=seed_val)
        loader = await cls.create(
            batch_size=hparams.batch_size,
            sequence_length=hparams.sequence_length,
            pages_info=pages,
            tokenizer=tokenizer,
            pack_samples=pack_samples
        )
        return loader, pages