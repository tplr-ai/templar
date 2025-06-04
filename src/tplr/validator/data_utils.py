# The MIT License (MIT)
# © 2025 tplr.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import TYPE_CHECKING, Dict

import tplr
from ..common.async_utils import retry_call
from ..utils.logging import log_with_context

if TYPE_CHECKING:
    from .validator_core import ValidatorCore


async def preload_dataloader(validator_instance: "ValidatorCore", seed: int) -> Dict:
    """
    Load pages and create a dataloader. Seed > 255 implies a "random" context.
    
    Args:
        validator_instance: ValidatorCore instance
        seed: Seed for data loading. Seeds > 255 are considered "random" context
        
    Returns:
        Dictionary containing loader, pages, and is_random flag
    """
    is_random = seed > 255
    
    log_with_context(
        "debug",
        f"Preloading dataloader with seed {seed} ({'random' if is_random else 'uid-specific'} context)",
        sync_window=validator_instance.sync_window,
        current_window=validator_instance.current_window,
        seed=seed,
        is_random=is_random,
    )
    
    # Load pages using retry mechanism
    pages = await retry_call(
        tplr.r2_dataset.R2DatasetLoader.next_pages,
        offset=validator_instance.sync_window * validator_instance.hparams.pages_per_window,
        n_pages=validator_instance.hparams.pages_per_window,
        seed=seed,
        attempts=3,
        delay=1,
        context=f"loading pages with seed {seed}",
    )
    
    if pages is None:
        log_with_context(
            "error",
            f"Failed to load pages with seed {seed}",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
            seed=seed,
        )
        return {"loader": None, "pages": None, "is_random": is_random}
    
    # Create dataloader using retry mechanism
    loader = await retry_call(
        tplr.r2_dataset.R2DatasetLoader.create,
        batch_size=validator_instance.hparams.batch_size,
        sequence_length=validator_instance.hparams.sequence_length,
        pages_info=pages,
        tokenizer=validator_instance.tokenizer,
        attempts=3,
        delay=1,
        context=f"creating dataloader with seed {seed}",
    )
    
    if loader is None:
        log_with_context(
            "error",
            f"Failed to create dataloader with seed {seed}",
            sync_window=validator_instance.sync_window,
            current_window=validator_instance.current_window,
            seed=seed,
        )
        return {"loader": None, "pages": pages, "is_random": is_random}
    
    log_with_context(
        "debug",
        f"Successfully preloaded dataloader with seed {seed}",
        sync_window=validator_instance.sync_window,
        current_window=validator_instance.current_window,
        seed=seed,
        is_random=is_random,
        pages_count=len(pages) if pages else 0,
    )
    
    return {"loader": loader, "pages": pages, "is_random": is_random} 