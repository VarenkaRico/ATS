
import json
import os
from langdetect import detect, DetectorFactory
from xhtml2pdf import pisa
import re
from io import BytesIO

import boto3
from botocore.exceptions import NoCredentialsError

import streamlit as st
# ==== AWS CONFIG ====

from utils.aws_clients import s3, S3_BUCKET

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


def boldify(text: str) -> str:
    """
    Converts Markdown-style **bold** to <strong>bold</strong> in HTML.
    """
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)

def generate_clean_html(
    sections: dict,
    role: str = "",
    location: str = "",
    logo_url: str = "",
    skip_sections: list = None,
    title: str = "Job Description"
) -> str:
    skip_sections = set(skip_sections or [])

    html_sections = ""
    for section, content in sections.items():
        if section in skip_sections:
            continue

        content_formatted = boldify(content.replace('\n', '<br>'))

        html_sections += f"""
        <div class="section">
            <h2>{section}</h2>
            <p>{content_formatted}</p>
        </div>
        """

    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Helvetica, Arial, sans-serif;
                font-size: 11pt;
                color: #222;
                margin: 50px;
                line-height: 1.6;
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 3px solid #eee;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .header h1 {{
                letter-spacing: 4px;
                font-size: 18pt;
                text-transform: uppercase;
                font-weight: normal;
            }}
            .header img {{
                height: 50px;
            }}
            h2 {{
                font-size: 13pt;
                margin-top: 25px;
                color: #444;
                border-bottom: 1px solid #ccc;
                padding-bottom: 4px;
            }}
            p {{
                margin: 8px 0;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{title}</h1>
            {"<img src='" + logo_url + "' alt='Logo'>" if logo_url else ""}
        </div>

        {f"<p><strong>Role:</strong> {role}<br><strong>Location:</strong> {location}</p>" if role or location else ""}
        {html_sections}
    </body>
    </html>
    """
    return html


def generate_pdf_from_html(html: str) -> BytesIO:
    output = BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=output)
    if pisa_status.err:
        raise Exception("PDF generation failed")
    output.seek(0)
    return output

