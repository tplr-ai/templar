import boto3
from botocore.exceptions import ClientError

def delete_all_objects_in_bucket(endpoint_url, access_key, secret_key):
    """
    Delete every object in the specified bucket, derived from the endpoint URL.
    
    :param endpoint_url: The R2 endpoint URL, e.g. https://<ACCOUNT_ID>.r2.cloudflarestorage.com
    :param access_key: Access key for R2
    :param secret_key: Secret key for R2
    """
    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='auto'
    )

    try:
        # Extract bucket name from endpoint URL
        bucket_name = endpoint_url.split('//')[1].split('.')[0]

        paginator = s3.get_paginator('list_objects_v2')
        objects_to_delete = []

        for page in paginator.paginate(Bucket=bucket_name):
            if 'Contents' in page:
                objects_to_delete.extend([{'Key': obj['Key']} for obj in page['Contents']])

        if not objects_to_delete:
            print("Bucket is already empty.")
            return

        # Delete objects in batches (S3 limit is 1000 per request)
        batch_size = 1000
        for i in range(0, len(objects_to_delete), batch_size):
            batch = objects_to_delete[i : i + batch_size]
            response = s3.delete_objects(
                Bucket=bucket_name,
                Delete={'Objects': batch}
            )
            if 'Errors' in response:
                for error in response['Errors']:
                    print(f"Error deleting {error['Key']}: {error['Message']}")

        print(f"Successfully deleted {len(objects_to_delete)} objects from the bucket.")

    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"Error deleting objects: {error_code}")
    except IndexError:
        print("Could not parse the bucket name from the endpoint URL.")

# Example usage
if __name__ == "__main__":
    endpoint_url = 'https://88799e91c4a200e4056b45fe84df622c.r2.cloudflarestorage.com'
    access_key = '21e7419c5b2dbeb56f8cd0bd2742feff'
    secret_key = 'fa5fc9e6a9f3b0b8d5fa44efa3c9bf7b0c003f2cf3b6b7bba84d8adf4e8edeae'
    delete_all_objects_in_bucket(endpoint_url, access_key, secret_key)