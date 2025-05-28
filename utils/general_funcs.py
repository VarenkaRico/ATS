
import json
import os
from langdetect import detect, DetectorFactory

import boto3
from botocore.exceptions import NoCredentialsError

import streamlit as st
# ==== AWS CONFIG ====

session = boto3.Session(profile_name=os.getenv("AWS_PROFILE", "default"))
print(f"Using AWS profile: {os.getenv('AWS_PROFILE', 'default')}")
#session = boto3.Session(profile_name="recruitment-assistant")
s3 = session.client("s3")
bedrock = session.client("bedrock-runtime")

#S3_BUCKET = 'recruitment-agent-vrnk'
S3_BUCKET = os.getenv("S3_BUCKET", "default")
# ==== JOB DESCRIPTION STEP ====
MODEL_ID = "amazon.nova-lite-v1:0"

# ==== S3 UPLOAD ====
def upload_to_s3(file, key_name):
    try:
        s3.upload_fileobj(file, S3_BUCKET, key_name)
        return f"Uploaded successfully to {S3_BUCKET}/{key_name}"
    except NoCredentialsError:
        return "Credentials not available"
    except Exception as e:
        return f"Upload failed: {e}"


def load_chunks_from_s3(object_id, type):
    if type == "resume":
        s3_key = "resumes/resumes.json"
    elif type == "job description":
        s3_key = "job_descriptions/job_descriptions.json"

    try:
        s3_object = s3.get_object(Bucket = S3_BUCKET, Key = s3_key)
        job_data = json.loads(s3_object["Body"].read().decode("utf-8"))
        if object_id in job_data:
            return job_data[object_id]["chunks"]
        
        else:
            raise ValueError(f"{type} ID '{object_id}' not found in {s3_key}")
    
    except Exception as e:
        st.error(f"❌ Error loading job chunks: {e}")
        return []
    
def detect_language(text):
    """
    Detect the language of a given text using langdetect.

    Args:
        text (str): The parsed resume text.

    Returns:
        str: Detected language code (e.g., 'en', 'es', 'fr') or 'unknown'.
    """
    try:
        if isinstance(text, tuple):
            text = text[1]  # Extract actual string if passed as tuple
        return detect(text.strip()) if len(text.strip()) > 50 else "unknown"
    except:
        return "unknown"

