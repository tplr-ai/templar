#!/usr/bin/env python3

import re
import argparse
import os
import sys
import boto3
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
from botocore.config import Config as BotoConfig
from urllib.parse import urlparse
from tplr.logging import logger


def create_r2_client(aws_access_key_id, aws_secret_access_key, endpoint_url):
    try:
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        config = BotoConfig(signature_version="v4")
        s3_client = session.client("s3", config=config, endpoint_url=endpoint_url)

        logger.info("Successfully created R2 client")
        return s3_client

    except ClientError as e:
        logger.error(f"Failed to create R2 client: {str(e)}")
        sys.exit(1)


def get_endpoint_url(account_id):
    try:
        # Get R2 storage endpoint URL based on account ID
        return f"https://{account_id}.r2.cloudflarestorage.com"
    except Exception as e:
        logger.error(f"Failed to construct endpoint URL: {e}")
        sys.exit(1)


def is_valid_endpoint(endpoint_url):
    try:
        parsed = urlparse(endpoint_url)
        if not all([parsed.scheme, parsed.netloc]):
            return False
        # Additional checks can be added here if necessary
        return True
    except Exception as e:
        logger.error(f"Failed to validate endpoint URL: {e}")
        return False


def confirm_action(action_type, scope):
    """Seek user confirmation before performing an action.

    Args:
        action_type (str): Type of action ('delete' or 'wipe').
        scope (str): Describes the scope of the action.

    Returns:
        bool: True if confirmed, False otherwise.
    """
    while True:
        response = (
            input(f"\nAre you sure you want to {action_type} {scope}? [y/N] ")
            .strip()
            .lower()
        )
        if response == "y":
            return True
        elif response == "n":
            return False
        else:
            print("Invalid input. Please respond with 'y' for yes or 'n' for no.")


def is_older_than(hours, dt):
    """
    Checks if the given datetime `dt` is older than `hours` hours ago.

    Args:
        hours (int): Number of hours to check against.
        dt (datetime): Datetime object to compare.

    Returns:
        bool: True if `dt` is older than `hours` hours, False otherwise.
    """
    cutoff = datetime.now(dt.tzinfo) - timedelta(hours=hours)
    return dt < cutoff


def delete_chunk(s3_client, bucket_name, chunk):
    try:
        print(f"Attempting to delete {len(chunk)} objects...")
        response = s3_client.delete_objects(  # noqa: F841
            Bucket=bucket_name,
            Delete={
                "Objects": chunk  # Ensure chunk is a list of {'Key': '...'} dicts
            },
        )
        print(f"Successfully deleted {len(chunk)} objects.")
    except ClientError as e:
        print(f"Error deleting chunk: {e}")
        raise


def delete_older_objects(s3_client, bucket_name, hours):
    """
    Deletes objects from an S3 bucket that are older than a specified number of hours.

    Args:
        s3_client (boto3.client): Boto3 client for S3.
        bucket_name (str): Name of the S3 bucket.
        hours (int): Number of hours. Objects last modified older than this will be deleted.
    """
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name)

        keys_to_delete = []

        for page in page_iterator:
            contents = page.get("Contents", [])

            if not contents:
                break  # No more objects to process

            for obj in contents:
                key = obj["Key"]
                last_modified = obj["LastModified"]

                if is_older_than(hours, last_modified):
                    print(f"Marking '{key}' for deletion.")
                    keys_to_delete.append({"Key": key})

                # Delete in chunks of 1000
                if len(keys_to_delete) >= 1000:
                    # Second confirmation before deleting the current batch
                    confirm_delete = input(
                        f"You are about to delete {len(keys_to_delete)} files. Proceed? (yes/No): "
                    )
                    if not confirm_delete.lower().startswith("y"):
                        print("Deletion aborted by user.")
                        exit()

                    delete_chunk(s3_client, bucket_name, keys_to_delete)
                    keys_to_delete = []

        # Delete any remaining objects after the loop
        if keys_to_delete:
            # Second confirmation before deleting the current batch
            confirm_delete = input(
                f"You are about to delete {len(keys_to_delete)} files. Proceed? (yes/No): "
            )
            if not confirm_delete.lower().startswith("y"):
                print("Deletion aborted by user.")
                exit()

            print(f"Deleting remaining {len(keys_to_delete)} objects...")
            delete_chunk(s3_client, bucket_name, keys_to_delete)

    except ClientError as e:
        print(f"Error listing or deleting objects: {e}")
        raise


def delete_objects_with_prefix(s3_client, bucket_name, prefix):
    """Delete objects that start with the specified prefix."""
    try:
        # Confirm before proceeding
        scope = f"delete objects with prefix '{prefix}' in bucket '{bucket_name}'"
        if not confirm_action("delete", scope):
            logger.info(
                f"Deletion of objects with prefix '{prefix}' in bucket '{bucket_name}' cancelled by user."
            )
            return

        paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        keys_to_delete = []

        for page in page_iterator:
            contents = page.get("Contents", [])

            if not contents:
                break  # No more objects to process

            for obj in contents:
                keys_to_delete.append({"Key": obj["Key"]})

                # Delete in chunks of 1000
                if len(keys_to_delete) >= 1000:
                    # Second confirmation before deleting the current batch
                    confirm_delete = input(
                        f"You are about to delete {len(keys_to_delete)} files. Proceed? (yes/No): "
                    )
                    if not confirm_delete.lower().startswith("y"):
                        print("Deletion aborted by user.")
                        exit()

                    delete_chunk(s3_client, bucket_name, keys_to_delete)
                    keys_to_delete = []

            # Delete any remaining objects after the loop
        if keys_to_delete:
            # Second confirmation before deleting the current batch
            confirm_delete = input(
                f"You are about to delete {len(keys_to_delete)} files. Proceed? (yes/No): "
            )
            if not confirm_delete.lower().startswith("y"):
                print("Deletion aborted by user.")
                exit()

            print(f"Deleting remaining {len(keys_to_delete)} objects...")
            delete_chunk(s3_client, bucket_name, keys_to_delete)
    except ClientError as e:
        logger.error(f"Failed to delete objects with prefix: {e}")


def delete_objects_with_suffix(s3_client, bucket_name, suffix):
    """Delete objects that end with a specified suffix."""
    try:
        # Confirm before proceeding
        if not suffix:
            logger.error("Error: No suffix specified.")
            return

        scope = f"delete objects ending with '{suffix}' in bucket '{bucket_name}'"
        if not confirm_action("delete", scope):
            logger.info(
                f"Deletion of objects ending with '{suffix}' in bucket '{bucket_name}' cancelled by user."
            )
            return

        pattern = re.compile(re.escape(suffix) + "$")

        paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name)

        keys_to_delete = []

        for page in page_iterator:
            contents = page.get("Contents", [])

            if not contents:
                break  # No more objects to process

            for obj in contents:
                if pattern.search(obj["Key"]):
                    keys_to_delete.append({"Key": obj["Key"]})

                # Delete in chunks of 1000
                if len(keys_to_delete) >= 1000:
                    # Second confirmation before deleting the current batch
                    confirm_delete = input(
                        f"You are about to delete {len(keys_to_delete)} files. Proceed? (yes/No): "
                    )
                    if not confirm_delete.lower().startswith("y"):
                        print("Deletion aborted by user.")
                        exit()

                    delete_chunk(s3_client, bucket_name, keys_to_delete)
                    keys_to_delete = []

            # Delete any remaining objects after the loop
        if keys_to_delete:
            # Second confirmation before deleting the current batch
            confirm_delete = input(
                f"You are about to delete {len(keys_to_delete)} files. Proceed? (yes/No): "
            )
            if not confirm_delete.lower().startswith("y"):
                print("Deletion aborted by user.")
                exit()

            print(f"Deleting remaining {len(keys_to_delete)} objects...")
            delete_chunk(s3_client, bucket_name, keys_to_delete)
    except ClientError as e:
        logger.error(f"Failed to delete objects with suffix: {e}")


def wipe_bucket(s3_client, bucket_name):
    """Delete all objects in the specified bucket."""
    try:
        # Confirm before proceeding
        scope = f"wipe all objects in bucket '{bucket_name}'"
        if not confirm_action("wipe", scope):
            logger.info(f"Wiping of bucket '{bucket_name}' cancelled by user.")
            return

        # 2nd Confirm before proceeding
        scope = "Your data will be PERMANENTLY DELETED. Are you sure?"
        if not confirm_action("wipe", scope):
            logger.info(
                f"Wiping of bucket '{bucket_name}' cancelled by user at 2nd confirmation."
            )
            return

        paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name)

        total_deleted = 0

        for page in page_iterator:
            if "Contents" not in page:
                continue

            chunk = []
            for obj in page["Contents"]:
                chunk.append({"Key": obj["Key"]})

                # Delete in chunks of 1000
                if len(chunk) >= 1000:
                    delete_chunk(s3_client, bucket_name, chunk)
                    chunk = []
                    total_deleted += len(chunk)

            # Delete any remaining objects after the loop
            if chunk:
                delete_chunk(s3_client, bucket_name, chunk)
                total_deleted += len(chunk)

        logger.info(f"Deleted {total_deleted} objects in bucket '{bucket_name}'.")
    except ClientError as e:
        logger.error(f"Failed to wipe bucket: {e}")


def main():
    parser = argparse.ArgumentParser(description="Manage Cloudflare R2 bucket")
    group_actions = parser.add_argument_group("Actions (at least one required)")

    group_actions.add_argument(
        "--delete-old",
        type=int,
        help="Delete objects older than X hours; defaults to 48 if not specified",
    )
    group_actions.add_argument(
        "--prefix", type=str, help="Delete objects with this prefix"
    )
    group_actions.add_argument(
        "--suffix", type=str, help="Delete objects that end with this suffix string"
    )
    group_actions.add_argument(
        "--wipe-bucket",
        action="store_true",
        help="Delete all objects in the specified bucket",
    )

    group_config = parser.add_argument_group("Configuration (optional)")

    group_config.add_argument(
        "--miner-bucket",
        action="store_true",
        help="Use R2_Gradients_* variables to access bucket",
    )
    group_config.add_argument(
        "--dataset-bucket",
        action="store_true",
        help="Use R2_Dataset_* variables to access bucket",
    )
    group_config.add_argument(
        "--bucket-name", type=str, help="Specify a custom bucket name"
    )
    group_config.add_argument(
        "--access-key",
        type=str,
        help="Specify an access key instead of environment variables",
    )
    group_config.add_argument(
        "--secret-key",
        type=str,
        help="Specify a secret key instead of environment variables",
    )
    group_config.add_argument(
        "--account-id",
        type=str,
        help="Specify an account ID instead of environment variables",
    )
    group_config.add_argument(
        "--endpoint-url",
        type=str,
        help="Specify an endpoint URL instead of constructing it from the account ID",
    )

    args = parser.parse_args()

    # Check if at least one action is provided
    if (
        args.delete_old is None
        and args.prefix is None
        and args.suffix is None
        and not args.wipe_bucket
    ):
        logger.error("Error: No action specified. Must specify at least one action.")
        parser.print_help()
        sys.exit(1)

    # Determine configuration parameters
    if args.miner_bucket:
        config = "GRADIENTS"
    elif args.dataset_bucket:
        config = "DATASET"
    else:
        config = None

    if not args.bucket_name and config:
        bucket_name = os.environ.get(f"R2_{config.upper()}_BUCKET_NAME")
        access_key = os.environ.get(f"R2_{config.upper()}_WRITE_ACCESS_KEY_ID")

        secret_key = os.environ.get(f"R2_{config.upper()}_WRITE_SECRET_ACCESS_KEY")
        account_id = os.environ.get(f"R2_{config.upper()}_ACCOUNT_ID")
        endpoint_url = (
            args.endpoint_url or f"https://{account_id}.r2.cloudflarestorage.com"
        )
    else:
        bucket_name = args.bucket_name
        access_key = args.access_key or os.getenv("R2_GRADIENTS_WRITE_ACCESS_KEY_ID")
        secret_key = args.secret_key or os.getenv(
            "R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY"
        )
        account_id = args.account_id or os.getenv("R2_GRADIENTS_ACCOUNT_ID")
        endpoint_url = (
            args.endpoint_url or f"https://{account_id}.r2.cloudflarestorage.com"
        )

    # Configure S3 session
    s3_client = create_r2_client(access_key, secret_key, endpoint_url)

    if args.delete_old:
        hours = args.delete_old or 48
        delete_older_objects(s3_client, bucket_name, hours)
    if args.prefix:
        delete_objects_with_prefix(s3_client, bucket_name, args.prefix)
    if args.suffix:
        delete_objects_with_suffix(s3_client, bucket_name, args.suffix)
    if args.wipe_bucket:
        wipe_bucket(s3_client, bucket_name)


if __name__ == "__main__":
    main()
