from dotenv import load_dotenv
import os
load_dotenv()

import boto3
from botocore.exceptions import ClientError

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)

bucket = os.getenv('S3_BUCKET_NAME')
try:
    s3.head_bucket(Bucket=bucket)
    print(f'S3 bucket exists and is accessible: {bucket}')
except ClientError as e:
    print(f'Error: {e}')

# Write a test object
s3.put_object(
    Bucket=bucket,
    Key='test/connection_test.txt',
    Body=b'CTA pipeline S3 connection test'
)
print(f'Test write successful: s3://{bucket}/test/connection_test.txt')
print()
print('AWS setup complete.')