#!/usr/bin/env python3
# The MIT License (MIT)
# © 2024 templar.tech

import os
import sys
import asyncio
from dotenv import load_dotenv
from aiobotocore.session import get_session
from tplr import logger

# Add parent directory to path to import tplr
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tplr


async def cleanup_bucket():
    """Delete all objects in the R2 bucket"""
    # Load environment variables
    load_dotenv()

    # Validate required environment variables
    required_vars = [
        "R2_ACCOUNT_ID",
        "R2_WRITE_ACCESS_KEY_ID",
        "R2_WRITE_SECRET_ACCESS_KEY",
    ]

    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        logger.error(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
        sys.exit(1)

    # Get credentials from environment
    account_id = os.environ["R2_ACCOUNT_ID"]
    access_key_id = os.environ["R2_WRITE_ACCESS_KEY_ID"]
    secret_access_key = os.environ["R2_WRITE_SECRET_ACCESS_KEY"]

    # Initialize S3 client
    session = get_session()
    async with session.create_client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        region_name="enam",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=tplr.config.client_config,
    ) as client:
        logger.info("Listing objects in bucket...")

        # Use paginator to handle buckets with >1000 objects
        paginator = client.get_paginator("list_objects_v2")
        objects_to_delete = []

        try:
            async for page in paginator.paginate(Bucket=account_id):
                if "Contents" in page:
                    # Collect objects for deletion
                    objects_to_delete.extend(
                        [{"Key": obj["Key"]} for obj in page["Contents"]]
                    )

            if not objects_to_delete:
                logger.info("Bucket is already empty")
                return

            # Delete objects in batches of 1000 (S3 limit)
            batch_size = 1000
            for i in range(0, len(objects_to_delete), batch_size):
                batch = objects_to_delete[i : i + batch_size]
                logger.info(f"Deleting batch of {len(batch)} objects...")

                response = await client.delete_objects(
                    Bucket=account_id, Delete={"Objects": batch}
                )

                # Log any errors
                if "Errors" in response:
                    for error in response["Errors"]:
                        logger.error(
                            f"Error deleting {error['Key']}: {error['Message']}"
                        )

            logger.success(
                f"Successfully deleted {len(objects_to_delete)} objects from bucket"
            )

        except Exception as e:
            logger.error(f"Error cleaning bucket: {str(e)}")
            sys.exit(1)


if __name__ == "__main__":
    logger.info("Starting bucket cleanup...")
    asyncio.run(cleanup_bucket())
