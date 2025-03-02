import re
import sys
import time
import asyncio
import argparse
import threading

import requests
import bittensor as bt
import tplr
from tplr import __version__
from tplr.config import client_config, BUCKET_SECRETS
import botocore
from pprint import pprint
import torch
import itertools
from torch.nn.functional import cosine_similarity
import numpy as np
import os, json
CF_REGION_NAME: str = "enam"
WINDOW_OFFSET = 1

class Grafana:
    
    # Command line config items.
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='Miner script')
        parser.add_argument('--netuid', type=int, default=3, help='Bittensor network UID.')
        parser.add_argument('--project', type=str, default='templar', help='Wandb project.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
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

        self.bucket = self.comms.get_own_bucket('gradients', 'read')
        # self.comms.try_commit(self.wallet, self.bucket)
        # self.comms.fetch_commitments()
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
        self.start_window = None

        self.version = __version__
        self.version = self.get_tplr_version()

    async def initialize(self):
        # Start background block listener
        self.loop = asyncio.get_running_loop()
        self.listener = threading.Thread(
            target=self.block_listener,
            args=(self.loop,),
            daemon=True,
        ).start()
        self.comms.start_commitment_fetcher()
        self.comms.start_background_tasks()
        self.start_window, self.started_time = await self.get_start_window()
        tplr.logger.info(f"Using start_window: {self.start_window}")
        tplr.logger.info(f"Started Time: {self.started_time}")
        self.comms.commitments = await self.comms.get_commitments()
        
        # Save started time and started window and show it.
        
        
        # self.comms.start_background_tasks()

    def get_tplr_version(self):
        url = "https://raw.githubusercontent.com/tplr-ai/templar/main/src/tplr/__init__.py"

        response = requests.get(url)

        if response.status_code == 200:
            match = re.search(r'__version__ = ["\']([^"\']+)["\']', response.text)
            if match:
                return match.group(1)  # ✅ Extracts version number
            else:
                print("Version not found.")
        else:
            print("Failed to fetch file.")
    
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
                filename = f"gradient-{window}-{uid}-v{self.version}.pt"
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
        self.del_old_grad_dict(self.window_info, window)
        
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
    
    def get_avg_wnd_duration(self):
        window_keys = self.window_info.keys()
        count = len(window_keys)
        
        if count < 2:
            return None
        
        total_duration = 0
        for i, window in enumerate(window_keys):
            if i == 0: continue
            
            window_duration = self.window_info[int(window)]["window_time"] - self.window_info[int(window) - 1]["window_time"]
            total_duration += window_duration
        
        total_duration = total_duration / i
        return total_duration
            
    
    def get_metagraph_info(self):
        """
        Synchronizes the metagraph and retrieves relevant information.

        Returns:
            dict: A dictionary containing key details about the metagraph state.
        """
        self.metagraph.sync()  # Synchronize the metagraph to fetch the latest state

        # Retrieve the required metagraph attributes
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

    async def run(self):
        await self.initialize()

        self.grad_dict = {}
        step_window = self.current_window - WINDOW_OFFSET

        print(f"\n{'-' * 20} Window: {step_window} {'-' * 20}")
        gradients = {}
        gradients_metadata = {}

        while True:
            # Check and update the step_window if it has changed
            if step_window != self.current_window - WINDOW_OFFSET:
                global_step = step_window - self.start_window
                
                active_peers = self.grad_dict.get(step_window, [])
                error_peers = self.grad_error_dict.get(step_window, [])
                
                print(f"\n{'-' * 20} Window: {step_window}, Global Step: {global_step} {'-' * 20}")
                # print(self.get_metagraph_info())

                avg_wnd_duration = self.get_avg_wnd_duration()
                print(f"Average Window Duration: {avg_wnd_duration}")
                
                active_uids = [peer["uid"] for peer in active_peers]
                active_uids.sort()
                print(f"window: {step_window}, Length of Actives: {len(active_uids)} Active list: {active_uids}")
                
                self.comms.update_peers_with_buckets()
                self.peers = self.comms.peers
                print(f"Gather Peers: {self.peers}")
                
                active_miners = self.comms.eval_peers
                diff_miners = []
                for uid in active_miners:
                    if uid not in active_uids:
                        diff_miners.append(uid)
                
                # Real Active miners is eval peers.
                print(f"Total evaluation peers: {len(self.comms.eval_peers)}")
                
                print(f"Total diff miners: {len(diff_miners)}")
                print(f"Diff Miners: {diff_miners}")
                
                
                # for peer in active_peers:
                #     print(f"Active Peer Data: {peer}")
                
                # SAVE THIS METADATA TO DB AND SHOW IN GRAFANA!
                downloaded_uids = [int(uid) for uid in gradients.keys()]
                downloaded_uids.sort()
                print(f"Downloaded Length : {len(downloaded_uids)}, UIDs: {downloaded_uids}")
                # for uid in gradients.keys():
                #     print(f"Metadata for UID {uid}: {gradients_metadata[uid]}")

                similarities = await self.compute_cosine_similarities(gradients)

                # Print Similarity Matrix
                # await self.print_similarity_matrix(similarities)
                
                # SAVE THIS LIST TO DB AND SHOW IN GRAFANA!
                bad_peers = await self.analyze_similarities(similarities, active_peers, window=step_window, threshold=0.99)
                
                step_window = self.current_window - WINDOW_OFFSET
                gradients = {}
                gradients_metadata = {}
                
                with open("window_info.txt", "w") as file:
                    file.write(str(step_window))
                
            # Initialize the list for the current window if not already present
            self.grad_dict.setdefault(step_window, [])
            self.grad_error_dict.setdefault(step_window, [])

            # Retrieve and update active miners for the current window
            active_miners, error_miners = await self.get_active_miners(step_window)
            self.grad_dict[step_window].extend(active_miners)
            self.grad_error_dict[step_window].extend(error_miners)
            
            active_peers = self.grad_dict.get(step_window, [])
            download_uids = []
            for peer in active_peers:
                if peer["uid"] not in gradients.keys():
                    download_uids.append(peer["uid"])
            
            # Download gradients
            num_samples = min(7, len(download_uids))  # Ensure we don’t exceed available elements
            download_uids = np.random.choice(download_uids, size=num_samples, replace=False)
            
            result_gradients, result_metadata = await self.download_gradients(download_uids, step_window, key="gradient")
            gradients.update(result_gradients)
            gradients_metadata.update(result_metadata)
            
            await asyncio.sleep(0.1)

    # Listens for new blocks and sets self.current_block and self.current_window
    def block_listener(self, loop):
        def handler(event):
            self.current_block = int(event["header"]["number"])  # type: ignore
            new_window = int(self.current_block / self.hparams.blocks_per_window)
            if new_window != self.current_window:
                self.current_window = new_window
                self.comms.current_window = self.current_window
                tplr.logger.info(
                    f"New block received. Current window updated to: {self.current_window}"
                )
                self.window_info[new_window] = {}
                self.window_info[new_window]["window_time"] = time.time()

        while not self.stop_event.is_set():
            try:
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(
                    handler
                )
            except Exception as e:
                tplr.logger.error(f"Block subscription error: {e}")
                time.sleep(1)
                
    async def download_gradients(self, uids, window, key, timeout=10, local=False, stale_retention=100):
        """Downloads gradients asynchronously for multiple UIDs with speed optimization and dtype fix."""
        
        async def fetch(uid):
            result = await self.comms.get_with_retry(uid, window, key, timeout, local, stale_retention)
            if result is not None:
                state_dict, _ = result  # Unpack the tuple
                if "metadata" in state_dict.keys():
                    metadata = state_dict['metadata']
                    state_dict.pop("metadata", None)
                else:
                    metadata = None
                
                return uid, {k: torch.as_tensor(v, dtype=torch.float32) for k, v in state_dict.items()}, metadata
            return None, None, None

        tasks = [fetch(uid) for uid in uids]
        results = await asyncio.gather(*tasks)

        # Filter out None results and construct dictionary
        gradient_tensors = {uid: data for uid, data, _ in results if uid is not None}
        gradient_metadata = {uid: metadata for uid, _, metadata in results if uid is not None}

        return gradient_tensors, gradient_metadata
        
    async def print_similarity_matrix(self, similarities):
        """Prints cosine similarity matrix in the console in a readable format."""
        uids = sorted(set(uid for pair in similarities.keys() for uid in pair))
        uid_index = {uid: idx for idx, uid in enumerate(uids)}

        # Initialize similarity matrix with zeros
        similarity_matrix = [[0.0] * len(uids) for _ in range(len(uids))]

        # Fill the matrix with similarity values
        for (uid1, uid2), similarity in similarities.items():
            i, j = uid_index[uid1], uid_index[uid2]
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity  # Ensure symmetry

        # Print Header Row (UIDs)
        header = "     " + "  ".join(f"{uid:5}" for uid in uids)
        print(header)
        print("-" * len(header))

        # Print each row
        for i, uid in enumerate(uids):
            row_values = "  ".join(f"{similarity_matrix[i][j]:.2f}" for j in range(len(uids)))
            print(f"{uid:5} | {row_values}")

        
    async def compute_cosine_similarities(self, gradients):
        """Computes cosine similarities asynchronously in parallel for better performance."""
        similarity_results = {}

        async def compute_pair(uid1, grads1_dict, uid2, grads2_dict):
            """Compute cosine similarity between two gradient sets."""
            total_similarity = 0
            count = 0

            for key in grads1_dict.keys():
                if key in grads2_dict:
                    tensor1 = grads1_dict[key].flatten()
                    tensor2 = grads2_dict[key].flatten()
                    
                    if tensor1.shape == tensor2.shape:
                        similarity = cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0)).item()
                        total_similarity += similarity
                        count += 1

            avg_similarity = total_similarity / count if count > 0 else 0.0
            return (uid1, uid2), avg_similarity

        # Ensure correct unpacking of gradients.items()
        tasks = [
            compute_pair(uid1, grads1, uid2, grads2)
            for (uid1, grads1), (uid2, grads2) in itertools.combinations(gradients.items(), 2)
        ]

        results = await asyncio.gather(*tasks)

        for (pair, similarity) in results:
            similarity_results[pair] = similarity

        return similarity_results

    async def analyze_similarities(self, similarities, active_peers, window, threshold=0.99):
        """
        Analyzes cosine similarities to detect bad peers.
        
        If similarity between two peers exceeds the threshold, the later peer 
        (based on timestamp) is regarded as a bad peer.
        
        Args:
            similarities (dict): { (uid1, uid2): similarity_value, ... }
            active_peers (list): List of active peers with their timestamps.
            window (int): Current window number.
            threshold (float): Threshold above which peers are considered similar.
        
        Output:
            Prints candidates for bad peers and selected bad peers.
            Saves bad peers with window info to a file.
        
        Returns:
            list: List of bad peers detected.
        """
        # Create a mapping of UID -> timestamp
        peer_timestamps = {peer["uid"]: peer["timestamp"] for peer in active_peers}

        candidates = []
        bad_peers = set()

        # Check each similarity pair
        for (uid1, uid2), similarity in similarities.items():
            if similarity >= threshold:
                timestamp1 = peer_timestamps.get(uid1, "Unknown")
                timestamp2 = peer_timestamps.get(uid2, "Unknown")

                # Determine the later peer (newer timestamp = bad peer)
                if timestamp1 != "Unknown" and timestamp2 != "Unknown":
                    if timestamp1 > timestamp2:
                        bad_peer = uid1
                    else:
                        bad_peer = uid2
                    bad_peers.add(bad_peer)

                    candidates.append((uid1, uid2, similarity, bad_peer))

        # Print detected candidates
        print("\n=== Candidate Bad Peers (Similarity > {:.2f}) ===".format(threshold))
        if candidates:
            for uid1, uid2, similarity, bad_peer in candidates:
                print(f"⚠️  {uid1} and {uid2} have similarity {similarity:.2f}. Potential bad peer: {bad_peer}")
        else:
            print("✅ No bad peer candidates detected.")

        # Print final selected bad peers
        print("\n=== Selected Bad Peers ===")
        if bad_peers:
            print("🚨 Bad Peers:", ", ".join(map(str, bad_peers)))
        else:
            print("✅ No bad peers detected.")

        # Save bad peers to file
        await self.save_bad_peers_to_file(window, list(bad_peers))

        return list(bad_peers)


    async def save_bad_peers_to_file(self, window, bad_peers):
        """
        Saves bad peers information to a file in append mode.

        Args:
            window (int): The current window number.
            bad_peers (list): List of bad peers detected.
        """
        file_path = "bad_peers_log.json"
        
        entry = [window, self.version, {"bad peers": bad_peers}]
        
        with open(file_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        print(f"📁 Bad peers for window {window} appended to {file_path}")

    async def get_start_window(self) -> int:
        while True:
            try:
                (
                    validator_bucket,
                    validator_uid,
                ) = await self.comms._get_highest_stake_validator_bucket()
                if validator_bucket is None:
                    tplr.logger.warning(
                        "No highest staked validator bucket found. Retrying in 10 seconds."
                    )
                    await asyncio.sleep(10)
                    continue

                tplr.logger.info(
                    f"Attempting to fetch start_window from UID {validator_uid} bucket {validator_bucket.name}"
                )

                started_time = ""
                async with self.comms.session.create_client(
                    "s3",
                    endpoint_url=self.comms.get_base_url(validator_bucket.account_id),
                    region_name=CF_REGION_NAME,
                    config=client_config,
                    aws_access_key_id=validator_bucket.access_key_id,
                    aws_secret_access_key=validator_bucket.secret_access_key,
                ) as s3_client:
                    filename=f"start_window_v{self.version}.json"
                    tplr.logger.debug(
                        f"Checking for {filename} in bucket {validator_bucket.name}"
                    )
                    response = await s3_client.head_object(
                        Bucket=validator_bucket.name, Key=filename
                    )
                    # Get and log the created timestamp of the file
                    started_time = response["LastModified"].isoformat()
                        
                # Fetch 'start_window.json' using s3_get_object
                start_window_data = await self.comms.s3_get_object(
                    key=f"start_window_v{self.version}.json", bucket=validator_bucket
                )
                if start_window_data is not None:
                    # Check if start_window_data is already a dict
                    if isinstance(start_window_data, dict):
                        start_window_json = start_window_data
                    else:
                        # If it's bytes, decode and load JSON
                        start_window_json = json.loads(
                            start_window_data.decode("utf-8")
                        )

                    start_window = start_window_json["start_window"]
                    tplr.logger.info(f"Fetched start_window: {start_window}")
                    return start_window, started_time

                tplr.logger.warning(
                    "start_window.json not found or empty. Retrying in 10 seconds."
                )
                await asyncio.sleep(10)
            except Exception as e:
                tplr.logger.error(f"Error fetching start_window: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(Grafana().run())