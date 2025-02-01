import sys
import time
import asyncio
import argparse
import threading
import bittensor as bt
import tplr
from tplr import __version__
from tplr.config import client_config, BUCKET_SECRETS
import botocore
from pprint import pprint

CF_REGION_NAME: str = "enam"
WINDOW_OFFSET = 1

class Grafana:
    
    # Command line config items.
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='Miner script')
        parser.add_argument('--netuid', type=int, default=3, help='Bittensor network UID.')
        parser.add_argument('--project', type=str, default='templar', help='Wandb project.')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--trace', action='store_true', help='Enable trace logging')
        parser.add_argument('--peers', type=int, nargs='+', default=[], help='List of UIDs to peer with')
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        config = bt.config(parser)
        if config.debug:
            tplr.debug()
        if config.trace:
            tplr.trace()
        return config
    
    def __init__(self):
        tplr.logger.debug("Starting initialization...")
        
        # Init config and load hparams
        self.config = Grafana.config()
        self.hparams = tplr.load_hparams()
        
        # Init bittensor objects
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            tplr.logger.error(f'\n\t[bold]The wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}[/bold]')
            sys.exit()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        
        # Init comms
        self.comms = tplr.comms.Comms(
            wallet=self.wallet,
            save_location='/tmp',
            key_prefix='model',
            config=self.config,
            netuid=self.config.netuid,
            metagraph=self.metagraph,
            hparams=self.hparams,
            uid=self.uid,  
        )

        self.bucket = self.comms.get_own_bucket()
        # self.comms.try_commit(self.wallet, self.bucket) # is it necessary?
        self.comms.fetch_commitments()

        # Init state params
        self.stop_event = asyncio.Event()
        self.current_block = self.subtensor.block
        self.current_window = int(self.current_block / self.hparams.blocks_per_window)
        self.comms.current_window = self.current_window 
        self.step_counter = 0

        # Add step tracking
        self.global_step = 0
        self.window_step = 0
        
        # Gradient Checked Dictionary
        self.grad_dict = {}
        self.grad_error_dict = {}
        self.window_info = {}

    async def initialize(self):
        # Start background block listener
        self.loop = asyncio.get_running_loop()
        self.listener = threading.Thread(
            target=self.block_listener,
            args=(self.loop,),
            daemon=True,
        ).start()
        self.comms.start_commitment_fetcher()
        # self.comms.start_background_tasks()
    
    async def is_gradient_exist(self, uid: int, window: int):
        """Check if the miner has uploaded a gradient for a specific window and get the created timestamp of the file."""
        tplr.logger.debug(f"Checking if UID {uid} is active for window {window}")

        peer_bucket = self.comms.commitments.get(uid)
        if not peer_bucket:
            tplr.logger.debug(f"No bucket committed for UID {uid}")
            return False, None

        try:
            async with self.comms.session.create_client(
                "s3",
                endpoint_url=self.comms.get_base_url(peer_bucket.account_id),
                region_name=CF_REGION_NAME,
                config=client_config,
                aws_access_key_id=peer_bucket.access_key_id,
                aws_secret_access_key=peer_bucket.secret_access_key,
            ) as s3_client:
                filename = f"gradient-{window}-{uid}-v{__version__}.pt"
                tplr.logger.debug(
                    f"Checking for {filename} in bucket {peer_bucket.name}"
                )
                try:
                    response = await s3_client.head_object(
                        Bucket=peer_bucket.name, Key=filename
                    )
                    tplr.logger.debug(f"Found {filename} for UID {uid}")
                    # Get and log the created timestamp of the file
                    created_timestamp = response["LastModified"].isoformat()
                    content_length = response["ContentLength"]
                    tplr.logger.debug(
                        f"File {filename} for UID {uid} was created at {created_timestamp}"
                    )
                    
                    grad_info = {
                        "uid": int(uid),
                        "timestamp": created_timestamp, 
                        "content_length": content_length, 
                        "bucket_name": peer_bucket.account_id, 
                        "filename": filename
                    }
                    
                    return True, grad_info
                except botocore.exceptions.ClientError as e:
                    if e.response["Error"]["Code"] not in ["404"]:
                        tplr.logger.error(
                            f"Error checking activity for UID {uid}: {e}"
                        )
                        grad_info = {
                            "uid": int(uid),
                            "bucket_name": peer_bucket.account_id, 
                        }
                        return False, grad_info
                    tplr.logger.debug(f"{filename} not found for UID {uid}")
        except Exception as e:
            tplr.logger.error(f"Error accessing bucket for UID {uid}: {e}")
            return False, None

        return False, None
    
    def del_old_grad_dict(self, grad_dict: dict, current_window: int, retention=10):
        """
        Deletes old data from self.grad_dict, retaining only the last `retention` windows.
        
        Args:
            current_window (int): The current window number.
            retention (int): The number of recent windows to retain (default is 10).
        """
        # Calculate the cutoff window
        cutoff_window = current_window - retention

        # Delete all entries in grad_dict older than the cutoff window
        old_keys = [key for key in grad_dict if key < cutoff_window]
        for key in old_keys:
            del grad_dict[key]

        tplr.logger.debug(f"Deleted old data before window {cutoff_window}. Remaining windows: {list(grad_dict.keys())}")

    async def get_active_miners(self, window):
        """
        Retrieve a list of active miners and their timestamps for a specific window.
        """
        
        # Delete old grad dict
        self.del_old_grad_dict(self.grad_dict, window)
        self.del_old_grad_dict(self.grad_error_dict, window)
        
        # Get all UIDs and the ones already checked for the current window
        all_uids = set(self.comms.commitments.keys())
        checked_uids = {item['uid'] for item in self.grad_dict.get(window, [])} | \
                       {item['uid'] for item in self.grad_error_dict.get(window, [])}

        # Identify UIDs that haven't been checked yet
        unchecked_uids = all_uids - checked_uids
        
        # Asynchronously check for gradient existence for each unchecked UID
        if unchecked_uids:
            results = await asyncio.gather(
                *(self.is_gradient_exist(uid, window) for uid in unchecked_uids)
            )

            # Filter and structure active miners data
            active_miners = [
                grad_info for uid, (is_active, grad_info) in zip(unchecked_uids, results)
                if is_active
            ]
            
            error_miners = [
                grad_info for uid, (is_active, grad_info) in zip(unchecked_uids, results)
                if is_active == False and grad_info != None
            ]
        else:
            active_miners = []
            error_miners = []

        return active_miners, error_miners
    
    def get_grad_dict(self):
        return self.grad_dict, self.grad_error_dict
    
    def get_current_window(self):
        return self.current_window
    
    def get_grad_dict_window(self, window):
        return self.grad_dict[window], self.grad_error_dict[window]
    
    def get_window_info(self, window):
        return self.window_info[window]
    
    def get_metagraph_info(self):
        """
        Synchronizes the metagraph and retrieves relevant information.

        Returns:
            dict: A dictionary containing key details about the metagraph state.
        """
        self.metagraph.sync()  # Synchronize the metagraph to fetch the latest state

        # Retrieve the required metagraph attributes
        tplr.logger.info(f"\n{'-' * 20} metagraph: {self.metagraph} {'-' * 20}")
        metagraph_info = {
            "emission": self.metagraph.emission,
            "incentive": self.metagraph.incentive,
            "weights": self.metagraph.weights,
            "consensus": self.metagraph.consensus,
            "total_stake": self.metagraph.total_stake,
            "ranks": self.metagraph.ranks,
            "stake": self.metagraph.stake,
            "dividends": self.metagraph.dividends,
            "block": self.metagraph.block,
            "active": self.metagraph.active,
            "hotkeys": self.metagraph.hotkeys,
            "coldkeys": self.metagraph.coldkeys,
        }

        return metagraph_info

    # async def run(self):
    #     await self.initialize()

    #     self.grad_dict = {}
    #     step_window = self.current_window - WINDOW_OFFSET

    #     tplr.logger.info(f"\n{'-' * 20} Window: {step_window} {'-' * 20}")

    #     while True:
    #         # Check and update the step_window if it has changed
    #         if step_window != self.current_window - WINDOW_OFFSET:
    #             print(f"window: {step_window}, Active list: {self.grad_dict.get(step_window, [])}")
    #             print(f"window: {step_window}, Error list: {self.grad_error_dict.get(step_window, [])}")

    #             step_window = self.current_window - WINDOW_OFFSET
    #             tplr.logger.info(f"\n{'-' * 20} Window: {step_window} {'-' * 20}")
    #             self.comms.update_peers_with_buckets()

    #         # Initialize the list for the current window if not already present
    #         self.grad_dict.setdefault(step_window, [])
    #         self.grad_error_dict.setdefault(step_window, [])

    #         # Retrieve and update active miners for the current window
    #         active_miners, error_miners = await self.get_active_miners(step_window)
    #         self.grad_dict[step_window].extend(active_miners)
    #         self.grad_error_dict[step_window].extend(error_miners)

    #         await asyncio.sleep(1)

    # Listens for new blocks and sets self.current_block and self.current_window
    def block_listener(self, loop):
        def handler(event, _u, _s):
            self.current_block = int(event['header']['number'])
            new_window = int(self.current_block / self.hparams.blocks_per_window)
            if new_window != self.current_window:
                self.current_window = new_window
                self.comms.current_window = self.current_window  # Synchronize comms current_window
                self.window_info[new_window] = {}
                self.window_info[new_window]["window_time"] = time.time()
        while not self.stop_event.is_set():
            try:
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(handler)
                break
            except Exception:
                time.sleep(1)