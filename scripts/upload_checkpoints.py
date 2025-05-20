import os
import boto3
from botocore.exceptions import NoCredentialsError
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get credentials from environment variables
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY') 
AWS_REGION_NAME = os.environ.get('AWS_REGION_NAME')

# Log credentials status (safely)
logger.info(f"AWS Access Key ID exists: {bool(AWS_ACCESS_KEY_ID)}")
logger.info(f"AWS Secret Access Key exists: {bool(AWS_SECRET_ACCESS_KEY)}")
logger.info(f"AWS Region Name: {AWS_REGION_NAME}")
logger.info(f"S3 Bucket:")

def upload_to_s3(local_file, bucket, s3_file):
    """Upload a file to S3"""
    # Initialize S3 client with credentials
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION_NAME
    )

    try:
        # Get file size for progress bar
        file_size = os.path.getsize(local_file)
        
        # Create progress bar
        progress = tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading {local_file}")

        def callback(bytes_transferred):
            progress.update(bytes_transferred)

        logger.info(f"Starting upload of {local_file} to s3://{bucket}/{s3_file}")
        s3_client.upload_file(
            local_file, 
            bucket, 
            s3_file,
            Callback=callback
        )
        progress.close()
        logger.info("Upload Successful")
        return True
    except FileNotFoundError:
        logger.error(f"The file {local_file} was not found")
        return False
    except NoCredentialsError:
        logger.error("Credentials not available")
        return False

def main():
    # Set up paths and names
    local_file = "/workspace/xcodec/outputs/2025-05-19/22-02-58/logs/last.ckpt"
    s3_folder = "candy-ckpt"
    s3_filename = "candy-second.ckpt"
    s3_path = f"{s3_folder}/{s3_filename}"
    
    # Upload the file
    upload_to_s3(
        local_file=local_file,
        bucket="pixa-datasets",
        s3_file=s3_path
    )

if __name__ == "__main__":
    main()
