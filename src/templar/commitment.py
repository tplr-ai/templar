from substrateinterface import SubstrateInterface
from retry import retry
from typing import Optional, Dict
import bittensor as bt
from .logging import logger

def get_all_commitments(substrate: SubstrateInterface, netuid: int, metagraph, block: Optional[int] = None) -> Dict[int, str]:
    """
    Retrieves all commitments from the blockchain for a given network UID and maps them to bucket names.
    
    This function queries the blockchain for all commitments associated with a specific network UID,
    decodes the commitment data which contains S3 bucket information, and maps each validator's UID
    to their associated bucket name.

    Args:
        substrate (SubstrateInterface): Connection to the substrate blockchain
        netuid (int): Network UID to query commitments for
        metagraph: Bittensor metagraph object containing network state
        block (Optional[int]): Specific block number to query. If None, queries latest block

    Returns:
        Dict[int, str]: Mapping of validator UIDs to their S3 bucket names

    Example:
        >>> substrate = SubstrateInterface(url="wss://archivelnode.opentensor.ai:443")
        >>> metagraph = bt.metagraph(netuid=1)
        >>> buckets = get_all_commitments(substrate, 1, metagraph)
        >>> print(buckets[0])  # Prints bucket name for validator UID 0
    """
    @retry(delay=2, tries=3, backoff=2, max_delay=4)
    def query_commitments():
        # Get block hash if specific block requested, otherwise use latest
        block_hash = None if block is None else substrate.get_block_hash(block)
        logger.info(f"Querying commitments for netuid {netuid} at block {'latest' if block_hash is None else block_hash}")
        
        # Query the Commitments module's CommitmentOf storage
        return substrate.query_map(
            module='Commitments',
            storage_function='CommitmentOf',
            params=[netuid],
            block_hash=block_hash
        )

    # Execute query with retry logic and convert iterator to list
    result = list(query_commitments())

    # Create mapping of hotkey (SS58 address) to validator UID
    hotkey_to_uid = dict(zip(metagraph.hotkeys, metagraph.uids))
    buckets = {}

    # Process each commitment result
    for key, value in result:
        # Extract hotkey (SS58 address) from key
        hotkey = key.value

        # Skip if hotkey not in our network
        if hotkey not in hotkey_to_uid:
            continue

        # Get validator UID for this hotkey
        uid = hotkey_to_uid[hotkey]

        # Extract commitment hex data from nested structure
        # Commitment data is stored in fields[0] of the commitment info
        commitment_info = value.value.get('info', {})
        fields = commitment_info.get('fields', [])

        # Skip if commitment data is malformed
        if not fields or not isinstance(fields[0], dict):
            continue

        # Extract hex value from first field
        field_value = next(iter(fields[0].values()))
        # Remove '0x' prefix if present
        if field_value.startswith('0x'):
            field_value = field_value[2:]

        # Decode hex data to UTF-8 string
        try:
            commitment = bytes.fromhex(field_value).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to decode commitment for UID {uid}: {e}")
            continue

        # Extract bucket name from commitment string
        # Format is "bucket_name:additional_data"
        parts = commitment.split(':')
        if parts:
            bucket_name = parts[0]
            buckets[uid] = bucket_name
        else:
            logger.error(f"Failed to extract bucket name for UID {uid}: Commitment is empty")
            continue

    return buckets
