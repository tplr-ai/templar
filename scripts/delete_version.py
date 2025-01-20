#!/usr/bin/env python3

import os
import asyncio
from aiobotocore.session import get_session
from dotenv import load_dotenv
import botocore.config

# Load environment variables from .env file
load_dotenv()

# Get R2 credentials from environment variables
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID")
R2_BUCKET_NAME = R2_ACCOUNT_ID  # Bucket name is the same as account_id

R2_WRITE_ACCESS_KEY_ID = os.environ.get("R2_WRITE_ACCESS_KEY_ID")
R2_WRITE_SECRET_ACCESS_KEY = os.environ.get("R2_WRITE_SECRET_ACCESS_KEY")

if not all([R2_ACCOUNT_ID, R2_WRITE_ACCESS_KEY_ID, R2_WRITE_SECRET_ACCESS_KEY]):
    print("Missing required environment variables.")
    exit(1)

# Configuration for S3 client
client_config = botocore.config.Config(max_pool_connections=32)
CF_REGION_NAME = "enam"


def get_base_url(account_id):
    return f"https://{account_id}.r2.cloudflarestorage.com"


async def list_and_delete_checkpoints():
    session = get_session()
    endpoint_url = get_base_url(R2_ACCOUNT_ID)
    async with session.create_client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=CF_REGION_NAME,
        config=client_config,
        aws_access_key_id=R2_WRITE_ACCESS_KEY_ID,
        aws_secret_access_key=R2_WRITE_SECRET_ACCESS_KEY,
    ) as s3_client:
        paginator = s3_client.get_paginator("list_objects_v2")
        prefix = "checkpoint"
        checkpoints = []
        async for page in paginator.paginate(Bucket=R2_BUCKET_NAME, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                last_modified = obj["LastModified"]
                size = obj["Size"]

                # If it ends with "v0.2.11.pt" then delete
                if key.endswith("v0.2.11.pt"):
                    print(f"Deleting file: {key}")
                    await s3_client.delete_object(Bucket=R2_BUCKET_NAME, Key=key)
                else:
                    checkpoints.append(
                        {
                            "Key": key,
                            "LastModified": last_modified,
                            "Size": size,
                        }
                    )

        if checkpoints:
            # Sort by LastModified in descending order
            checkpoints.sort(key=lambda x: x["LastModified"], reverse=True)
            print("Remaining checkpoints:")
            for checkpoint in checkpoints:
                print(
                    f"{checkpoint['Key']} - "
                    f"LastModified: {checkpoint['LastModified']}, "
                    f"Size: {checkpoint['Size']}"
                )
        else:
            print("No checkpoints found or all matching checkpoints were deleted.")


if __name__ == "__main__":
    asyncio.run(list_and_delete_checkpoints())
