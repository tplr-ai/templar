# Global imports
import bittensor as bt
from pydantic import ValidationError

# Local imports
import templar as tplr
from templar.schemas import Bucket


class ChainManager:
    def __init__(self, subtensor: bt.Subtensor, wallet: bt.wallet, netuid: int):
        """Class used to get commits from and send commits to the blockchain.

        Args:
            subtensor: Subtensor network object.
            wallet: The wallet associated with the neuron committing the data.
            netuid: The unique identifier of the subnetwork.
        """
        self.subtensor = subtensor
        self.wallet = wallet
        self.netuid = netuid

    def commit(self) -> None:
        """Commits bucket configuration data to the subtensor network.

        This method prepares and commits bucket configuration data associated
        with the wallet to the subtensor network. The data includes:
        - Account ID: A string of fixed length 32 characters.
        - Access key ID: A string of fixed length 32 characters.
        - Secret access key: A string of variable length (up to 64 characters).

        The commitment process involves:
        - Fetching the required configuration details from the `tplr.config.BUCKET_SECRETS`
          dictionary.
        - Concatenating the account ID, access key ID, and secret access key
          into a single string, in this exact order.
        - Committing the concatenated data to the subtensor network using the
          configured `netuid` and wallet.

        **Note:** The order of concatenation (account ID, access key ID, secret
        access key) is critical for correct parsing when the data is retrieved.

        Logs provide visibility into the data type and structure before
        committing.

        Raises:
            Any exceptions that might arise from the subtensor network
            communication are propagated.
        """
        concatenated = (
            tplr.config.BUCKET_SECRETS["account_id"]
            + tplr.config.BUCKET_SECRETS["read"]["access_key_id"]
            + tplr.config.BUCKET_SECRETS["read"]["secret_access_key"]
        )
        self.subtensor.commit(self.wallet, self.netuid, concatenated)
        tplr.logger.info(
            f"Committed {type(concatenated)} data of type to the network: {concatenated}"
        )

    def get_commitment(self, uid: int) -> Bucket:
        """Retrieves and parses committed bucket configuration data for a given
        UID.

        This method fetches commitment data for a specific UID from the
        subtensor network and decodes it into a structured format. The
        retrieved data is split into the following fields:
        - Account ID: A string of fixed length 32 characters.
        - Access key ID: A string of fixed length 32 characters.
        - Secret access key: A string of variable length (up to 64 characters).

        The parsed fields are then mapped to an instance of the `Bucket` class.
        When initializing the Bucket object, the account ID is also used as the
        bucket name.

        The retrieval process involves:
        - Fetching the commitment data for the specified UID using the
          configured `netuid` from the subtensor network.
        - Splitting the concatenated string into individual fields based on
          their expected lengths and order.
        - Mapping the parsed fields to a `Bucket` instance.

        **Note:** The order of fields (bucket name, account ID, access key ID,
        secret access key) in the concatenated string is critical for accurate
        parsing.

        Args:
            uid: The UID of the neuron whose commitment data is being
                retrieved.

        Returns:
            Bucket: An instance of the `Bucket` class containing the parsed
                bucket configuration details.

        Raises:
            ValueError: If the parsed data does not conform to the expected
                format for the `Bucket` class.
            Exception: If an error occurs while retrieving the commitment data
                from the subtensor network.
        """
        try:
            concatenated = self.subtensor.get_commitment(self.netuid, uid)
            tplr.logger.success(f"Commitment fetched: {concatenated}")
        except Exception as e:
            raise Exception(f"Couldn't get commitment from uid {uid} because {e}")
        if len(concatenated) != 128:
            raise ValueError(
                f"Commitment '{concatenated}' is of length {len(concatenated)} but should be of length 128."
            )

        try:
            return Bucket(
                name=concatenated[:32],
                account_id=concatenated[:32],
                access_key_id=concatenated[32:64],
                secret_access_key=concatenated[64:],
            )
        except ValidationError as e:
            raise ValueError(f"Invalid data in commitment: {e}")
