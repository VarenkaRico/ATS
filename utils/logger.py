# utils/json_logger.py
import os
import json
from datetime import datetime

import boto3
from botocore.exceptions import NoCredentialsError

from dotenv import load_dotenv
from io import BytesIO

import streamlit as st

load_dotenv()

AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
S3_BUCKET = os.getenv("S3_BUCKET", "default")
S3_LOG_KEY = "logs/log.json"

# Setup AWS
session = boto3.Session(profile_name=AWS_PROFILE)
s3 = session.client("s3")


def log_event(level, message, context=None, auto_upload=True):

    """
    Appends a structured log entry to the in-memory session log and optionally uploads to S3.

    Args:
        level (str): Logging level (e.g., 'INFO', 'ERROR').
        message (str): Log message describing the event.
        context (dict, optional): Additional context or metadata. Defaults to None.
        auto_upload (bool, optional): If True, logs will be immediately uploaded to S3. Defaults to True.

    Outputs:
        - Appends entry to st.session_state["logs"].
        - Optionally clears logs after upload.
        - Uploads log entries to S3 if auto_upload is True.
    """

    entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level.upper(),
        "message": message,
        "context": context or {}
    }

    # Initialize log list in memory
    if "logs" not in st.session_state:
        st.session_state["logs"] = []

    st.session_state["logs"].append(entry)

    if auto_upload:
        upload_log_to_s3()
        st.session_state["logs"] = []


def upload_log_to_s3():
    """
    Uploads accumulated session logs from memory to an S3 object, appending to existing logs if present.

    Returns:
        str: Status message indicating success, no logs to upload, or failure.

    Raises:
        Any exceptions during the upload process are caught and returned as part of the message.
    """

    try:
        # Get existing S3 log
        try:
            s3_obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_LOG_KEY)
            existing_logs = json.loads(s3_obj["Body"].read().decode("utf-8"))
        except s3.exceptions.NoSuchKey:
            existing_logs = []

        # Get local in-memory logs
        new_logs = st.session_state.get("logs", [])

        if not new_logs:
            return "⚠️ No logs to upload."

        # Merge and upload
        merged = existing_logs + new_logs
        log_bytes = BytesIO(json.dumps(merged, indent=2, ensure_ascii=False).encode("utf-8"))
        s3.upload_fileobj(log_bytes, S3_BUCKET, S3_LOG_KEY)

        st.session_state["logs"] = []  # Clear after upload
        return f"✅ Uploaded {len(new_logs)} logs to S3."

    except Exception as e:
        return f"❌ Failed to upload logs: {e}"