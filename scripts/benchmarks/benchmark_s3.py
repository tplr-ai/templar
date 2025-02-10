import os
import time
import logging
import asyncio
from aiobotocore.session import get_session
from pathlib import Path
from botocore.config import Config
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path, override=True)

# Cloudflare R2 credentials and settings
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
CF_REGION_NAME = os.getenv("CF_REGION_NAME", "auto")

# Backblaze B2 credentials and settings
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APPLICATION_KEY = os.getenv("B2_APPLICATION_KEY")
B2_REGION = os.getenv("B2_REGION")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME")

# Optional settings
OBJECT_KEY = os.getenv("OBJECT_KEY", "test_120mb.bin")
ITERATIONS = int(os.getenv("ITERATIONS", "5"))
TIMEOUT = int(os.getenv("TIMEOUT", "60"))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# File data (120MB)
EXPECTED_SIZE = 120 * 1024 * 1024  # 120MB

# Generate or verify test file
def get_test_file_data():
    local_file_path = os.getenv("LOCAL_FILE_PATH", OBJECT_KEY)
    if not os.path.isfile(local_file_path) or os.path.getsize(local_file_path) != EXPECTED_SIZE:
        logging.info(f"Generating test file '{local_file_path}' with {EXPECTED_SIZE} bytes.")
        with open(local_file_path, "wb") as f:
            f.write(b"\0" * EXPECTED_SIZE)
    with open(local_file_path, "rb") as f:
        return f.read()

test_data = get_test_file_data()

# Cloudflare R2 S3 Client Setup
class R2Client:
    def __init__(self, account_id, access_key_id, secret_access_key, region_name, bucket_name):
        self.account_id = account_id
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region_name = region_name
        self.bucket_name = bucket_name
        self.session = get_session()

    def get_base_url(self):
        return f"https://{self.account_id}.r2.cloudflarestorage.com"

    async def get_object(self, key):
        async with self.session.create_client(
            "s3",
            endpoint_url=self.get_base_url(),
            region_name=self.region_name,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
        ) as s3_client:
            response = await s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return await response["Body"].read()

    async def put_object(self, key, data):
        async with self.session.create_client(
            "s3",
            endpoint_url=self.get_base_url(),
            region_name=self.region_name,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
        ) as s3_client:
            await s3_client.put_object(Bucket=self.bucket_name, Key=key, Body=data)

# Backblaze S3 Client Setup
class BackblazeClient:
    def __init__(self, key_id, application_key, region, bucket_name):
        self.key_id = key_id
        self.application_key = application_key
        self.region = region
        self.bucket_name = bucket_name
        self.endpoint_url = f"https://s3.{self.region}.backblazeb2.com"

    def create_client(self):
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.key_id,
            aws_secret_access_key=self.application_key,
            config=Config(signature_version="s3v4")
        )

    def get_object(self, client, key):
        return client.get_object(Bucket=self.bucket_name, Key=key)["Body"].read()

    def put_object(self, client, key, data):
        client.put_object(Bucket=self.bucket_name, Key=key, Body=data)

# Benchmark Function
async def benchmark_s3_object():
    r2_client = R2Client(R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, CF_REGION_NAME, R2_BUCKET_NAME)
    b2_client = BackblazeClient(B2_KEY_ID, B2_APPLICATION_KEY, B2_REGION, B2_BUCKET_NAME)

    # Cloudflare R2 PUT Benchmark
    logging.info("Starting Cloudflare R2 PUT benchmark")
    r2_put_times = []
    for i in range(ITERATIONS):
        start = time.perf_counter()
        await r2_client.put_object(OBJECT_KEY, test_data)
        elapsed = time.perf_counter() - start
        r2_put_times.append(elapsed)
        logging.info(f"R2 PUT iteration {i+1}: {elapsed:.2f} seconds")

    r2_avg_put_time = sum(r2_put_times) / len(r2_put_times)
    logging.info(f"R2 PUT average time: {r2_avg_put_time:.2f} seconds")

    # Backblaze S3 PUT Benchmark
    logging.info("Starting Backblaze S3 PUT benchmark")
    b2_client_instance = b2_client.create_client()
    b2_put_times = []
    for i in range(ITERATIONS):
        start = time.perf_counter()
        b2_client.put_object(b2_client_instance, OBJECT_KEY, test_data)
        elapsed = time.perf_counter() - start
        b2_put_times.append(elapsed)
        logging.info(f"Backblaze S3 PUT iteration {i+1}: {elapsed:.2f} seconds")

    b2_avg_put_time = sum(b2_put_times) / len(b2_put_times)
    logging.info(f"Backblaze S3 PUT average time: {b2_avg_put_time:.2f} seconds")

    # Cloudflare R2 GET Benchmark
    logging.info("Starting Cloudflare R2 GET benchmark")
    r2_get_times = []
    for i in range(ITERATIONS):
        start = time.perf_counter()
        await r2_client.get_object(OBJECT_KEY)
        elapsed = time.perf_counter() - start
        r2_get_times.append(elapsed)
        logging.info(f"R2 GET iteration {i+1}: {elapsed:.2f} seconds")

    r2_avg_get_time = sum(r2_get_times) / len(r2_get_times)
    logging.info(f"R2 GET average time: {r2_avg_get_time:.2f} seconds")

    # Backblaze S3 GET Benchmark
    logging.info("Starting Backblaze S3 GET benchmark")
    b2_get_times = []
    for i in range(ITERATIONS):
        start = time.perf_counter()
        b2_client.get_object(b2_client_instance, OBJECT_KEY)
        elapsed = time.perf_counter() - start
        b2_get_times.append(elapsed)
        logging.info(f"Backblaze S3 GET iteration {i+1}: {elapsed:.2f} seconds")

    b2_avg_get_time = sum(b2_get_times) / len(b2_get_times)
    logging.info(f"Backblaze S3 GET average time: {b2_avg_get_time:.2f} seconds")

# Main
if __name__ == "__main__":
    asyncio.run(benchmark_s3_object())
