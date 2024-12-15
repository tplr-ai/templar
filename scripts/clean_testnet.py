import yaml
from pathlib import Path
import boto3
import shutil

import templar as tplr


def clean_r2_bucket():
    """Clean all checkpoint files from R2 bucket"""
    try:
        # Load credentials from .env.yaml
        env_path = Path(__file__).parent.parent / ".env.yaml"
        with open(env_path, "r") as f:
            config = yaml.safe_load(f)

        # Get R2 credentials
        account_id = config.get("account_id")
        write_creds = config.get("write", {})
        access_key_id = write_creds.get("access_key_id")
        secret_access_key = write_creds.get("secret_access_key")

        if not all([account_id, access_key_id, secret_access_key]):
            raise ValueError("Missing required R2 credentials in .env.yaml")

        # R2 connection settings
        session = boto3.Session(
            aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key
        )

        s3 = session.client(
            service_name="s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        )

        # List and delete all objects
        paginator = s3.get_paginator("list_objects_v2")

        deleted_count = 0
        for page in paginator.paginate(Bucket=account_id):
            if "Contents" in page:
                for obj in page["Contents"]:
                    s3.delete_object(Bucket=account_id, Key=obj["Key"])
                    deleted_count += 1

        tplr.logger.success(f"Deleted {deleted_count} files from R2 bucket")

    except FileNotFoundError:
        tplr.logger.error("Could not find .env.yaml file")
    except yaml.YAMLError:
        tplr.logger.error("Error parsing .env.yaml file")
    except Exception as e:
        tplr.logger.error(f"Error cleaning R2 bucket: {str(e)}")


def clean_local_folders():
    """Clean local wandb and checkpoints folders"""
    try:
        # Get project root directory
        root_dir = Path(__file__).parent.parent

        # Clean wandb
        wandb_dir = root_dir / "wandb"
        if wandb_dir.exists():
            shutil.rmtree(wandb_dir)
            tplr.logger.success("Cleaned wandb folder")

        # Clean checkpoints
        checkpoints_dir = root_dir / "checkpoints"
        if checkpoints_dir.exists():
            shutil.rmtree(checkpoints_dir)
            tplr.logger.success("Cleaned checkpoints folder")

    except Exception as e:
        tplr.logger.error(f"Error cleaning local folders: {str(e)}")


def main():
    """Main function to clean both R2 bucket and local folders"""
    tplr.logger.info("Starting cleanup process...")
    clean_r2_bucket()
    clean_local_folders()
    tplr.logger.success("Cleanup completed!")


if __name__ == "__main__":
    main()
