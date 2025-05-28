
import json
import os
import re

import boto3
from botocore.exceptions import NoCredentialsError

from datetime import datetime
import tempfile
from PyPDF2 import PdfReader

from io import BytesIO
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

def extract_resume_sections(resume_text):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": (
                        "You are an expert resume parser. Extract the following sections from the provided resume text:"
                        "\n- General information (summary, interests)"
                        "\n- Personal information (name, phone, email, websites)"
                        "\n- Experience"
                        "\n- Education/Studies"
                        "\n- Projects"
                        "\n- Skills"
                        "\n- Other (publications, patents, etc.)"
                        "\n\nReturn your response as a JSON object. If any section is missing, use 'NA'."
                        f"\n\nResume Text:\n{resume_text}"
                    )
                }
            ]
        }
    ]

    body = json.dumps({"messages": messages,
                       "inferenceConfig": {
                            "max_new_tokens": 5000
                        }
                    })

    try:
        response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json"
        )
        response_raw = response["body"].read().decode("utf-8").strip()
        response_body = json.loads(response_raw)
        content_text = response_body["output"]["message"]["content"][0]["text"]
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content_text, re.DOTALL)
        
        if match:
            json_string = match.group(1)

        else:
            json_string = content_text

        return json.loads(json_string)
    
    except json.JSONDecodeError as jde:
        return{"error": f"JSON decode error: {jde}"}
            
    except Exception as e:
        return {"error": f"Resume section extraction failed: {e}"}

def embed_resume_sections(sections_dict, bedrock_client, model_id="amazon.titan-embed-text-v2:0"):
    """
    Create embeddings for each section of a parsed resume.

    Args:
        sections_dict (dict): Parsed resume sections (e.g. from Streamlit session).
        bedrock_client: boto3 Bedrock client.
        model_id (str): Bedrock embedding model ID.

    Returns:
        dict: {
            section_name: {
                "text": original_text,
                "embedding": [vector]
            }, ...
        }
    """
    chunks_dict = {}

    for section, text in sections_dict.items():
        cleaned_text = text.strip()
        if not cleaned_text:
            continue

        try:
            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps({"inputText": cleaned_text}),
                contentType="application/json",
                accept="application/json"
            )
            result = json.loads(response["body"].read())
            embedding = result.get("embedding", [])

            chunks_dict[section] = {
                "text": cleaned_text,
                "embedding": embedding
            }

        except Exception as e:
            print(f"❌ Embedding failed for section '{section}': {e}")
            continue

    return chunks_dict

def flatten_section_content(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "\n".join(str(item) for item in content)
    elif isinstance(content, dict):
        return "\n".join(f"{k}: {v}" for k, v in content.items())
    else:
        return ""

# === FORMAT SECTION FOR DISPLAY ===
def format_section_for_display(section_data):
    if isinstance(section_data, str):
        return section_data
    elif isinstance(section_data, list):
        formatted_chunks = []
        for item in section_data:
            if isinstance(item, dict):
                formatted_chunks.append("\n".join(f"{k}: {v}" for k, v in item.items()))
            else:
                formatted_chunks.append(str(item))
        return "\n\n".join(formatted_chunks)
    elif isinstance(section_data, dict):
        return "\n".join(f"{k}: {v}" for k, v in section_data.items())
    else:
        return json.dumps(section_data, indent=2)

def save_resume_to_json(uuid_role, language, chunks, unique_id, s3_key="resumes/resumes.json"):
    """
    Save the resume and metadata directly to a JSON file in S3.

    Args:
        uuid_job_description: uuid of the job description to be matched to.
        language (str): Language of the resume.
        chunks (list[dict]): List of chunks with text and vector.
        file_name: uuid of the original file to easy identification
        s3_key (str): S3 key (path inside the bucket).
    """

    entry = {
        "role": uuid_role,
        "language": language,
        "date_created": datetime.now().isoformat(),
        "chunks": chunks,
        "file": unique_id
    }

    try:
        # Try downloading existing JSON from S3
        try:
            s3_object = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
            data = json.loads(s3_object["Body"].read().decode("utf-8"))
        except s3.exceptions.NoSuchKey:
            data = {}

        # Append new entry
        data[unique_id] = entry

        # Upload new JSON directly to S3
        json_bytes = BytesIO(json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))
        s3.upload_fileobj(json_bytes, S3_BUCKET, s3_key)

        return f"✅ Resume saved with ID {unique_id}"
    except NoCredentialsError:
        return "❌ AWS credentials not available"
    except Exception as e:
        return f"❌ Failed to upload JSON to S3: {e}"
    
def parse_resume_to_text(file_bytes):
    # Placeholder logic — replace with resume parser or Glue/EMR call
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            reader = PdfReader(tmp.name)
            text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

            return (f"Parsed content (preview):\n{text[:1000]}", text)
    except Exception as e:
        return f"Could not parse resume: {e}"