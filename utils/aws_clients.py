# utils/aws_clients.py
import os
import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
S3_BUCKET = os.getenv("S3_BUCKET", "default")

# Initialize session
session = boto3.Session(profile_name=AWS_PROFILE)

# Reusable clients
s3 = session.client("s3")
bedrock = session.client("bedrock-runtime")

# Model IDs
MODEL_NOVA = "amazon.nova-lite-v1:0"
MODEL_TITAN_EMBED = "amazon.titan-embed-text-v2:0"
