#!/usr/bin/env python3
# The MIT License (MIT)
# © 2025 tplr.ai

# ruff: noqa
# type: ignore

import os
import sys
import asyncio
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from aiobotocore.session import get_session
import botocore

# Find and load the correct .env file
env_path = Path(__file__).parent.parent / "docker" / ".env"
if not env_path.exists():
    print(f"Error: .env file not found at {env_path}")
    sys.exit(1)
load_dotenv(env_path, override=True)


async def test_read_access(client, bucket: str, prefix: str = "test_read_") -> bool:
    """Test read access by listing objects"""
    try:
        response = await client.list_objects_v2(Bucket=bucket, MaxKeys=1)
        print(f"{prefix}✅ Read access verified - Can list objects")
        return True
    except botocore.exceptions.ClientError as e:
        error_code = e.response["Error"]["Code"]
        print(f"{prefix}❌ Read access failed - {error_code}")
        return False


async def test_write_access(client, bucket: str, prefix: str = "test_write_") -> bool:
    """Test write access by creating and deleting a test object"""
    test_key = "test_permissions_file"
    test_content = b"test content"

    try:
        # Try to upload
        await client.put_object(Bucket=bucket, Key=test_key, Body=test_content)
        print(f"{prefix}✅ Write access verified - Can upload objects")

        # Clean up
        await client.delete_object(Bucket=bucket, Key=test_key)
        print(f"{prefix}✅ Write access verified - Can delete objects")
        return True
    except botocore.exceptions.ClientError as e:
        error_code = e.response["Error"]["Code"]
        print(f"{prefix}❌ Write access failed - {error_code}")
        return False


async def validate_credentials():
    """Validate both read and write credentials for gradients and dataset buckets"""
    required_vars = [
        "R2_GRADIENTS_ACCOUNT_ID",
        "R2_GRADIENTS_READ_ACCESS_KEY_ID",
        "R2_GRADIENTS_READ_SECRET_ACCESS_KEY",
        "R2_GRADIENTS_WRITE_ACCESS_KEY_ID",
        "R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY",
        "R2_DATASET_ACCOUNT_ID",
        "R2_DATASET_READ_ACCESS_KEY_ID",
        "R2_DATASET_READ_SECRET_ACCESS_KEY",
        "R2_DATASET_WRITE_ACCESS_KEY_ID",
        "R2_DATASET_WRITE_SECRET_ACCESS_KEY",
    ]

    # Check for missing environment variables
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        sys.exit(1)

    session = get_session()
    client_config = botocore.config.Config(
        max_pool_connections=32, retries={"max_attempts": 3}
    )

    # Test Gradients bucket
    print("\n🔍 Testing Gradients Bucket Access:")
    gradients_account_id = os.getenv("R2_GRADIENTS_ACCOUNT_ID")
    gradients_endpoint = f"https://{gradients_account_id}.r2.cloudflarestorage.com"

    # Test read credentials
    print("\n📖 Testing READ credentials:")
    async with session.create_client(
        "s3",
        endpoint_url=gradients_endpoint,
        region_name="auto",
        aws_access_key_id=os.getenv("R2_GRADIENTS_READ_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_GRADIENTS_READ_SECRET_ACCESS_KEY"),
        config=client_config,
    ) as read_client:
        can_read = await test_read_access(read_client, gradients_account_id, "  ")
        can_write = await test_write_access(read_client, gradients_account_id, "  ")
        if can_write:
            print("  ⚠️  WARNING: READ credentials have write access!")

    # Test write credentials
    print("\n✍️  Testing WRITE credentials:")
    async with session.create_client(
        "s3",
        endpoint_url=gradients_endpoint,
        region_name="auto",
        aws_access_key_id=os.getenv("R2_GRADIENTS_WRITE_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY"),
        config=client_config,
    ) as write_client:
        can_read = await test_read_access(write_client, gradients_account_id, "  ")
        can_write = await test_write_access(write_client, gradients_account_id, "  ")
        if not can_read:
            print("  ⚠️  WARNING: WRITE credentials should have read access!")

    # Test Dataset bucket
    print("\n🔍 Testing Dataset Bucket Access:")
    dataset_account_id = os.getenv("R2_DATASET_ACCOUNT_ID")
    dataset_endpoint = f"https://{dataset_account_id}.r2.cloudflarestorage.com"

    # Test read credentials
    print("\n📖 Testing READ credentials:")
    async with session.create_client(
        "s3",
        endpoint_url=dataset_endpoint,
        region_name="auto",
        aws_access_key_id=os.getenv("R2_DATASET_READ_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_DATASET_READ_SECRET_ACCESS_KEY"),
        config=client_config,
    ) as read_client:
        can_read = await test_read_access(read_client, dataset_account_id, "  ")
        can_write = await test_write_access(read_client, dataset_account_id, "  ")
        if can_write:
            print("  ⚠️  WARNING: READ credentials have write access!")

    # Test write credentials
    print("\n✍️  Testing WRITE credentials:")
    async with session.create_client(
        "s3",
        endpoint_url=dataset_endpoint,
        region_name="auto",
        aws_access_key_id=os.getenv("R2_DATASET_WRITE_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_DATASET_WRITE_SECRET_ACCESS_KEY"),
        config=client_config,
    ) as write_client:
        can_read = await test_read_access(write_client, dataset_account_id, "  ")
        can_write = await test_write_access(write_client, dataset_account_id, "  ")
        if not can_read:
            print("  ⚠️  WARNING: WRITE credentials should have read access!")


if __name__ == "__main__":
    print("🔐 Starting R2 Access Validation")
    asyncio.run(validate_credentials())
