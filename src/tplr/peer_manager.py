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

import asyncio
from collections import defaultdict

from tplr import logger
from tplr import __version__
from tplr.storage import StorageManager

import tempfile


class PeerManager:
    """Manages peer discovery and tracking"""

    def __init__(self, chain, hparams, metagraph):
        self.chain = chain
        self.hparams = hparams
        self.metagraph = metagraph
        self.active_peers = set()
        self.inactive_peers = set()
        self.eval_peers = defaultdict(int)

    async def track_active_peers(self):
        """Background task to keep track of active peers"""
        while True:
            active_peers = set()
            tasks = []
            semaphore = asyncio.Semaphore(10)  # Limit concurrent S3 requests

            logger.debug(f"Commitments: {self.chain.commitments}")

            async def check_peer(uid):
                async with semaphore:
                    is_active = await self.is_miner_active(
                        uid, recent_windows=self.hparams.recent_windows
                    )
                    if is_active:
                        active_peers.add(uid)

            for uid in self.chain.commitments.keys():
                tasks.append(check_peer(uid))

            await asyncio.gather(*tasks)
            self.active_peers = active_peers

            logger.info(
                f"Updated active peers: {[int(uid) for uid in self.active_peers]}"
            )

            # Update chain's active peers list
            self.chain.active_peers = self.active_peers

            # Update peer lists based on active peers
            self.chain.update_peers_with_buckets()

            await asyncio.sleep(self.hparams.active_check_interval)

    async def is_miner_active(self, uid: int, recent_windows: int = 3) -> bool:
        """Check if a miner is active by looking for recent gradient files"""
        # Get the bucket for this UID
        bucket = self.chain.get_bucket(uid)
        if not bucket:
            return False

        # Check the most recent windows
        current_window = self.chain.current_window
        for window in range(current_window, current_window - recent_windows, -1):
            if window <= 0:
                continue

            try:
                temp_dir = tempfile.TemporaryDirectory().name
                storage = StorageManager(temp_dir=temp_dir, save_location=temp_dir)

                gradient_key = f"gradient-{window}-{uid}-v{__version__}.pt"
                exists = await storage.s3_head_object(key=gradient_key, bucket=bucket)

                if exists:
                    return True
            except Exception as e:
                logger.error(f"Error checking activity for UID {uid}: {e}")

        return False
