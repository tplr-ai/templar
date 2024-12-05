# The MIT License (MIT)
# © 2024 templar.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# fmt: off

import os
import boto3
import time
import torch
import asyncio
import aiofiles
import botocore
import tempfile
from typing import List, Dict
from aiobotocore.session import get_session

import tplr as tplr

BUCKET = 'decis'
S3 = boto3.client('s3')
CLIENT_CONFIG = botocore.config.Config(max_pool_connections=256,)

class Comms:
    
    def __init__(self, subtensor, metagraph):
        ...
        
    def sync():
        ...

async def put(state_dict: dict, uid: str, window: int, key: str):
    tplr.logger.info(f"PUT {uid}/{window}/{key} -->")
    # Save the object to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as temp_file:
        torch.save(state_dict, temp_file)
        temp_file_path = temp_file.name
    # Upload the object to S3 asynchronously
    session = get_session()
    async with session.create_client(
        's3',
        region_name='us-east-1',
        config=CLIENT_CONFIG,
        aws_access_key_id='',
        aws_secret_access_key=''
        ) as s3_client:
            with open(temp_file_path, "rb") as f:
                await s3_client.put_object(Bucket=BUCKET, Key=f"{uid}/{window}/{key}", Body=f)
    # Delete tmp file
    os.remove(temp_file_path)
    tplr.logger.info(f"PUT {uid}/{window}/{key} <--")

    
async def get( uid: int, window: int, key: str, timeout: int = 30 ):
    full_key = f"{uid}/{window}/{key}"
    tplr.logger.info(f"GET {full_key} -->")
    try:
        # Create a temporary file to store the downloaded object
        with tempfile.NamedTemporaryFile(delete=True, suffix='.pt') as temp_file:
            temp_file_path = temp_file.name
            # Download the object with a timeout
            session = get_session()
            async with session.create_client(
                's3',
                region_name='us-east-1',
                config=CLIENT_CONFIG,
                aws_access_key_id='',
                aws_secret_access_key=''
            ) as s3_client:
                response = await asyncio.wait_for(
                    s3_client.get_object(Bucket=BUCKET, Key=f"{full_key}"),
                    timeout=timeout
                )
                async with aiofiles.open(temp_file_path, "wb") as outfile:
                    while True:
                        chunk = await response["Body"].read(1 * 1024 * 1024)
                        if not chunk:break
                        await outfile.write(chunk)
            # Load the object into memory
            with open(temp_file_path, 'rb') as f:
                state_dict = torch.load(f, weights_only=True)
        tplr.logger.info(f"GET {full_key} <--")
        return state_dict
    except asyncio.TimeoutError:
        tplr.logger.debug(f"Timeout occurred while downloading {full_key} from S3.")
        return None
    except Exception as e:
        tplr.logger.debug(f"An error occurred during GET: {full_key}: {e}")
        return None
    
async def get_with_retry(uid, window, key, timeout):
    """Attempt to get data from S3, retrying until success or timeout."""
    start_time = time.time()
    end_time = start_time + timeout
    while True:
        try:
            state_dict = await get(uid=uid, window=window, key=key)
            if state_dict == None:
                raise ValueError("wait...")
            return state_dict
        except Exception:
            if time.time() >= end_time:
                tplr.logger.info(f"GET {uid}/{window}/{key} x (timeout), {time.time()} > {int(end_time)}")
                return None
            await asyncio.sleep(0.1)  # Wait before retrying
    
async def gather( 
    state_dict: Dict[str, torch.Tensor], 
    my_uid: int, 
    uids: List[int], 
    window: int, 
    key:str, 
    timeout: int,
    device: str,
) -> List[ Dict[str, torch.Tensor] ]:
    # Put the object if exists.
    if state_dict != None:
        await put( 
            state_dict = state_dict, 
            uid = my_uid, 
            window = window, 
            key = key
        )
    # Create gather tasks for all other objects.
    gather_tasks = []
    for uid in uids:
        gather_tasks.append(
            get_with_retry(
                uid = uid,
                window = window,
                key = key,
                timeout = timeout
            )
        )        
    # Create buffer for responses.
    gather_result = {
        key: [ torch.zeros_like(value).to(device) for _ in uids ] for key, value in state_dict.items()
    }
    # Gather results async.
    responses = await asyncio.gather(*gather_tasks)
    # Fill results.
    for uid, resp in enumerate( responses ):
        if resp == None: continue
        for key in state_dict.keys():
            gather_result[ key ][ uid ] = resp[ key ].to(device)   
    # Return gather result.   
    return gather_result
                
    