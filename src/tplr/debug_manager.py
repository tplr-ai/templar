from typing import Optional
import json
from tplr import __version__


class DebugManager:
    """
    DebugManager service for retrieving debug data from the network.

    This service uses a Comms instance (provided during initialization)
    to access chain sync and storage functionalities.
    """

    def __init__(self, comms):
        """
        Initialize the DebugManager with a Comms instance.
        """
        self.comms = comms
        self.logger = comms.logger

    async def get_debug_dict(self, window: int) -> Optional[dict]:
        """
        Retrieve the debug dictionary for a specific window from the highest-staked validator bucket.
        """
        # Get the validator bucket and uid using the chain sync method.
        (
            validator_bucket,
            validator_uid,
        ) = await self.comms.chain._get_highest_stake_validator_bucket()
        if not validator_bucket or validator_uid is None:
            self.logger.warning(
                "No validator bucket available for debug data retrieval."
            )
            return None

        key = f"debug-{window}-{validator_uid}-v{__version__}.pt"
        self.logger.info(
            f"Attempting to retrieve debug dictionary for window {window} from validator {validator_uid}"
        )

        result = await self.comms.storage.s3_get_object(
            key=key, bucket=validator_bucket, timeout=20
        )
        if result is None:
            self.logger.warning(f"No debug dictionary found for window {window}")
            return None

        self.logger.info(f"Successfully retrieved debug dictionary for window {window}")
        try:
            # Assuming the debug data was stored as JSON bytes.
            if isinstance(result, bytes):
                decoded = result.decode("utf-8")
                debug_dict = json.loads(decoded)
                return debug_dict
            # Otherwise, return result directly.
            return result
        except Exception as e:
            self.logger.error(f"Error processing debug dictionary: {e}")
            return None
