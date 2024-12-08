# Global imports
import bittensor as bt
from retry import retry
from substrateinterface import SubstrateInterface
from typing import Optional, Dict

# Local imports
from .logging import logger
import templar as tplr
from templar.schemas import Bucket


def commit(subtensor: bt.Subtensor, wallet, netuid: int) -> None:
    """Commits bucket configuration data to the subtensor network.

    This method prepares and commits bucket configuration data to the subtensor network.
    The data includes:
    - Account ID: A string of fixed length 32 characters
    - Access key ID: A string of fixed length 32 characters
    - Secret access key: A string of variable length (up to 64 characters)

    The commitment process involves:
    - Concatenating the account ID, access key ID, and secret access key into a single string
    - Committing the concatenated data to the subtensor network using the provided netuid and wallet

    Args:
        subtensor: The subtensor network interface
        wallet: The wallet used to sign the commitment transaction
        netuid: The network UID to commit the data to

    Raises:
        Any exceptions from the subtensor network communication are propagated
    """
    concatenated = (
        tplr.config.BUCKET_SECRETS["account_id"]
        + tplr.config.BUCKET_SECRETS["read"]["access_key_id"]
        + tplr.config.BUCKET_SECRETS["read"]["secret_access_key"]
    )
    subtensor.commit(wallet, netuid, concatenated)
    logger.info(f"Committed data to the network: {concatenated}")


def get_all_commitments(
    substrate: SubstrateInterface,
    netuid: int,
    metagraph,
    block: Optional[int] = None,
) -> Dict[int, Bucket]:
    """Retrieves and parses all commitment data from the network for a given netuid.

    The commitment data for each neuron contains:
    - Access Key ID: 32 characters (positions 0-31)
    - Secret Access Key: 64 characters (positions 32-95)
    - Account ID: 32 characters (positions 96-127)
    Total length: 128 characters

    Args:
        substrate: The substrate interface for blockchain queries.
        netuid: Network UID to query commitments for.
        metagraph: Metagraph object containing hotkey->uid mappings.
        block: Optional block number to query at. If None, queries the latest block.

    Returns:
        Dict[int, Bucket]: Mapping of UIDs to their corresponding Bucket objects.
    """

    @retry(delay=2, tries=3, backoff=2, max_delay=4)
    def query_commitments():
        block_hash = None if block is None else substrate.get_block_hash(block)
        logger.info(
            f"Querying commitments for netuid {netuid} at block {'latest' if block_hash is None else block_hash}"
        )
        return substrate.query_map(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[netuid],
            block_hash=block_hash,
        )

    result = list(query_commitments())
    hotkey_to_uid = dict(zip(metagraph.hotkeys, metagraph.uids))
    commitments = {}

    for key, value in result:
        hotkey = key.value
        if hotkey not in hotkey_to_uid:
            continue

        uid = hotkey_to_uid[hotkey]
        commitment_info = value.value.get("info", {})
        fields = commitment_info.get("fields", [])

        if not fields or not isinstance(fields[0], dict):
            continue

        field_value = next(iter(fields[0].values()))
        if field_value.startswith("0x"):
            field_value = field_value[2:]

        try:
            concatenated = bytes.fromhex(field_value).decode("utf-8").strip()

            if len(concatenated) != 128:
                logger.debug(
                    f"Commitment '{concatenated}' has length {len(concatenated)}, expected 128."
                )
                continue

            # Use the suggested parsing logic
            bucket = Bucket(
                name=concatenated[:32],
                account_id=concatenated[:32],
                access_key_id=concatenated[32:64],
                secret_access_key=concatenated[64:],
            )
            commitments[uid] = bucket
            logger.debug(f"Bucket fetched and parsed for UID {uid}: {bucket.name}")

        except Exception as e:
            logger.error(f"Failed to decode and parse commitment for UID {uid}: {e}")
            continue

    return commitments
