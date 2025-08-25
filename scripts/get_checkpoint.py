# %%
VERSION = None

import time

import bittensor as bt

import tplr
from neurons import miner
from tplr import comms, hparams

# Get the metagraph
subtensor = bt.subtensor(network="finney")
metagraph = subtensor.metagraph(netuid=3)

staked = metagraph.S.tolist()
max_staked_uid = staked.index(max(staked))


# Shortcut config using miner's staticmethod
config = miner.Miner.miner_config()
config.netuid = 3

# hparams and comms manager
hparams_ = hparams.load_hparams("../hparams")
comms_mgr = comms.Comms(
    wallet=None,
    netuid=3,
    metagraph=metagraph,
    hparams=hparams_,
    config=config,
)

comms_mgr.start_commitment_fetcher()
bucket = None
while bucket is None:
    bucket = comms_mgr.get_bucket(uid=max_staked_uid)
    if bucket is not None:
        break
    print("Waiting for bucket...")
    time.sleep(2)

# Get checkpoint from latest or specified version
version = VERSION or tplr.__version__

# Latest checkpoint only
# latest_checkpoint = await comms_mgr._get_bucket_checkpoint(
#     bucket,
#     uid=max_staked_uid,
#     version=version,
# )

# List all checkpoints
checkpoints = await comms_mgr._list_bucket_checkpoints(
    bucket,
    uid=max_staked_uid,
    version=version,
)

# Approximately evenly spaced checkpoints to download
n_checkpoints = 4  # (there is off-by-one risk)
desired_checkpoints = list(checkpoints.keys())
desired_checkpoints = desired_checkpoints[:: (len(checkpoints) // n_checkpoints)]

# for ckpt in desired_checkpoints:
#     loaded_data = await self.s3_get_object(
#         key=checkpoints[ckpt], bucket=bucket
#     )

# %%
