# streamlit_recruitment_app.py
import streamlit as st
import boto3
import json
import tempfile
import os
from botocore.exceptions import NoCredentialsError
from PyPDF2 import PdfReader
import uuid
from datetime import datetime
from io import BytesIO
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import textwrap
import re
import time
import difflib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

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

# --- Generate job description ---
def generate_job_description(role, region, language):

    prompt = f"""
    You are an expert HR agent. 
    Generate a complete and inclusive job description for the role of '{role}'. 
    Include responsibilities, required qualifications, average range salary in {region} expressed in USD.
    
    Ensure the description:
    - Avoids bias related to gender, age, ethnicity, disability, or other protected characteristics.
    - Reflects modern inclusive recruiting standards.
    - Is written in **{language}**, including all section headers.
    - Follows the formatting and tone of the provided **Data Scientist** example — but only as a structure reference, not content.

    ---

    **Format Your Output Using This Structure** (keep headers in English if language is not English):
    - Job Title
    - Job Description
    - Key Responsibilities
    - Required Qualifications
    - Technical Skills
    - Soft Skills
    - Average Annual Salary Range in {region}
    - Equal Opportunity Statement
    - Application Process

    ---

    Use the following job description for a **Data Scientist** **only as a structural example** of formatting and tone. 
    Do not copy its responsibilities or qualifications. 
    Focus instead on how the content is presented.
    If a different language from english is selected, section headers should not be translated.

    ---

    **Example Template (DO NOT COPY CONTENT):**

    **Job Title:** Data Scientist

    **Job Description:**

    We are seeking a highly skilled and motivated Data Scientist to join our dynamic team. 
    The ideal candidate will have a passion for data analysis, machine learning, and statistical modeling. 
    This role involves working with large datasets to uncover insights and trends that will drive business decisions.

    **Key Responsibilities:**

    - *Data Analysis and Interpretation:* Analyze complex datasets to identify trends, patterns, and insights. Translate data findings into actionable business strategies.
    - *Model Development:* Develop and implement predictive models and machine learning algorithms to solve business problems.
    - *Data Visualization:* Create clear and compelling data visualizations and dashboards to communicate findings effectively to stakeholders.
    ...

    **Required Qualifications:**

    - *Education:* Bachelor’s degree in Data Science, Statistics, Computer Science, Mathematics, or a related field. A Master’s degree or Ph.D. is a plus.
    - *Experience:* Minimum of 3-5 years of experience in a data science role or a similar analytical position.
    ...
    
    **Technical Skills:**
    - Proficiency in programming languages such as Python, R, or SQL.
    - Experience with data visualization tools such as Tableau, Power BI, or similar.
    ...
    
    **Soft Skills:**
    - Excellent problem-solving skills and attention to detail.
    - Strong communication skills, both written and verbal.
    ...

    **Average Salary Range in {region} (USD):**

    - *Entry-Level:* $30,000 - $50,000 annually
    - *Mid-Level:* $50,000 - $100,000 annually
    - *Senior-Level:* $100,000 - $150,000+ annually

    *Note:* Salary ranges may vary based on location, company size, and specific industry.

    ---
    
    **Copy the following sections word-for-word if the language is English. Otherwise, translate them to {language}.**

    **Equal Opportunity Statement:**
    We are an equal opportunity employer and value diversity at our company. 
    We do not discriminate on the basis of race, religion, color, national origin, gender, sexual orientation, age, marital status, veteran status, or disability status.

    **Application Process:**
    Interested candidates are encouraged to submit their resume and cover letter detailing their relevant experience and skills. 
    We look forward to reviewing your application and potentially welcoming you to our team.
    """


    body = json.dumps({
        "inferenceConfig": {
            "max_new_tokens": 1000
        },
        "messages": [
            {     
                "role": "user",
                "content": [
                {
                    "text": prompt
                }
                ]
            }
        ]
    })
    try:
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            body=body,
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get('body').read().decode('utf-8'))
        return response_body["output"]["message"]["content"][0]["text"]
    except Exception as e:
        return f"Error: {e}"
    
#Get sections of the job description to facilitate matching
def split_job_description_sections(text):

    CANONICAL_HEADERS = [
    "Job Title",
    "Job Description",
    "Key Responsibilities",
    "Required Qualifications",
    "Technical Skills",
    "Soft Skills",
    "Average Annual Salary Range",
    "Equal Opportunity Statement",
    "Application Process"
]

    # Matches **Section Title:**
    pattern = r"\*\*(.*?)\*\*\s*:?[\r\n]+"
    parts = re.split(pattern, text)
    
    sections = {}
    if len(parts) < 2:
        return sections  # return empty if nothing matches

    # parts[0] is the text before the first header (often empty)
    for i in range(1, len(parts), 2):
        
        section_key = parts[i].strip().rstrip(":")
        section_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        #matched = difflib.get_close_matches(raw_section, CANONICAL_HEADERS, n=1, cutoff=0.6)
        #section_key = matched[0] if matched else raw_section
        sections[section_key] = section_text

    return sections

def get_section_embeddings_dict(dict_section):
    """
    Generate an embedding for each labeled section of text.

    Args:
        dict_sections (dict): {section_name: text}

    Returns:
        dict: {
            section_name: {
                "text": original_text,
                "embeding": [vector]
            }
        }, ...
    """

    dict_result = {}
    for section, text in dict_section.items():
        cleaned = text.strip()
        if not cleaned:
            continue

        try:
            print(f"Embedding section: {section} ({len(cleaned)} chars)")
            response = bedrock.invoke_model(
                modelId = "amazon.titan-embed-text-v2:0",
                body = json.dumps({"inputText":cleaned}),
                contentType = "application/json",
                accept="application/json"
            )

            result = json.loads(response["body"].read())

            dict_result[section] = {
                "text": cleaned,
                "embedding": result.get("embedding", [])
            }

        except Exception as e:
            print(f"Error embedding section '{section}':{e}")
            continue
    return dict_result

def save_job_description_to_json(role, region, language, chunks, s3_key="job_descriptions/job_descriptions.json"):
    """
    Save the job description and metadata directly to a JSON file in S3.

    Args:
        role (str): Role title.
        region (str): Geographic region.
        language (str): Language of the description.
        job_description (str): The generated text.
        chunks (list[dict]): List of chunks with text and vector.
        s3_key (str): S3 key (path inside the bucket).
    """
    unique_id = str(uuid.uuid4())
    entry = {
        "role": role,
        "region": region,
        "language": language,
        "date_created": datetime.now().isoformat(),
        "chunks": chunks
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

        return f"✅ Job description saved with ID {unique_id}"
    except NoCredentialsError:
        return "❌ AWS credentials not available"
    except Exception as e:
        return f"❌ Failed to upload JSON to S3: {e}"

# === Cluster CV Vs Vacante, texto CV una vez validada la cercanía de vectores ===

# ==== JOB DESCRIPTION VALIDATION ====
def validate_job_description(job_description, language):
    prompt = f"""
        You are a highly experienced Diversity & Inclusion HR auditor. Your goal is to critically evaluate job descriptions to ensure they do not unintentionally exclude qualified candidates.

        Analyze the following job description from the perspective of inclusive hiring and identify any element that could discourage capable individuals from applying.

        Specifically, assess:
        - Use of gendered, age-coded, or culturally biased language
        - In Spanish, look for masculine grammatical defaults (e.g., "el científico", "calificado") and suggest inclusive alternatives
        - Tone that may intimidate or alienate underrepresented groups
        - Unrealistic or rigid qualifications
        - Assumptions about background, education, or physical ability
        - Lack of flexibility (e.g., no mention of remote options, accommodations)
        - Any structural or phrasing issues that may reduce perceived accessibility

        Use a chain-of-thought approach to reason through the analysis.

        Return your output as a valid JSON object in the following structure:
        {{
        "result": "Satisfactory" | "Satisfactory with issues" | "Unsatisfactory",
        "reasoning": "Step-by-step reasoning for your conclusion",
        "main_issues_identified": ["Issue 1", "Issue 2", ...],
        "recommendations": ["Suggestion 1", "Suggestion 2", ...]
        }}

        Job Description to Review:
            {job_description}
    """

    body = json.dumps({
        "inferenceConfig": {
            "max_new_tokens": 2000
        },
        "messages": [
            {     
                "role": "user",
                "content": [
                {
                    "text": prompt
                }
                ]
            }
        ]
    })
    try:
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            body=body,
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get('body').read().decode('utf-8'))
        result_text = response_body["output"]["message"]["content"][0]["text"]

    except Exception as e:
        return f"Error: {e}"
    
    match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', result_text)
    if match:
        result_text = match.group(1)

    try:
        validation_result = json.loads(result_text)
    
    except json.JSONDecodeError:
        validation_result = {"result": "Unkown", "reasoning": "Model output could not be parsed", "main_issues_identified": [], "recommendations": []}

    return validation_result


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

# === 

# ==== BACKEND: Resume Parsing Placeholder ====
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
        "file": file_name
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
    
def detect_resume_language(text):
    """
    Detect the language of a given text using langdetect.

    Args:
        text (str): The parsed resume text.

    Returns:
        str: Detected language code (e.g., 'en', 'es', 'fr') or 'unknown'.
    """
    # try:
    #     # Clean short or noisy content
    #     clean_text = text.strip().replace('\n', ' ')
    #     if len(clean_text) < 50:
    #         return "unknown"
    #     return detect(clean_text)
    # except LangDetectException:
    #     return "unknown"
    try:
        if isinstance(text, tuple):
            text = text[1]  # Extract actual string if passed as tuple
        return detect(text.strip()) if len(text.strip()) > 50 else "unknown"
    except:
        return "unknown"

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

# === BACKEND: Job Description and Resumes matching
def load_job_descriptions_list(s3_key="job_descriptions/job_descriptions.json"):
    """
    Load list of job descriptions from S3 for dropdown selection.

    Returns:
        dict: {uuid: {'label': ..., 'data': ...}}
    """
    try:
        s3_object = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        all_jobs = json.loads(s3_object["Body"].read().decode("utf-8"))

        job_map = {}
        for uid, entry in all_jobs.items():
            label = f"{entry['role']} - {entry['region']} ({entry['date_created'][:10]})"
            job_map[uid] = {
                "label": label,
                "data": entry
            }
        return job_map
    except Exception as e:
        st.warning(f"❌ Could not load job descriptions: {e}")
        return {}

def get_resumes_for_job(job_description_id, s3_key = "resumes/resumes.json"):
    """
    Retrieve all resumes that were uploaded for a specific job description ID.

    Args:
        job_description_id (str): The UUID of the job description.
        s3_key (str): Path to the resumes JSON file in S3.

    Returns:
        dict: {resume_id: resume_data, ...} only for resumes matching the job_description_id
    """

    try:
        # Load the full resume database from S3
        s3_obj = s3.get_object(Bucket = S3_BUCKET,
                                      Key = s3_key)
        
        all_resumes = json.loads(s3_obj["Body"].read().decode("utf-8"))

        #Filter resumes by job description ID
        filtered = {
            resume_id: data
            for resume_id, data in all_resumes.items()
                if data.get("role") == job_description_id
        }
        
        return filtered
    
    except s3.exceptions.NoSuchKey:
        print("No resume file found.")
        return {}
    
    except Exception as e:
        print(f"Error loading resumes: {e}")
        return {}

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
        #st.error(f"❌ Error loading job chunks: {e}")
        print(f"❌ Error loading job chunks: {e}")
        return []
    
def get_info_for_job(job_description_id, type):
    """
    Retrieve all resumes that were uploaded for a specific job description ID.

    Args:
        job_description_id (str): The UUID of the job description.
        s3_key (str): Path to the resumes JSON file in S3.

    Returns:
        dict: {resume_id: resume_data, ...} only for resumes matching the job_description_id
    """

    if type == "resumes":
        s3_key ="resumes/resumes.json"

    elif type == "job description":
        s3_key = "job_descriptions/job_descriptions.json"

    try:
        # Load the full resume database from S3
        s3_obj = s3.get_object(Bucket = S3_BUCKET,
                                      Key = s3_key)
        
        all_info = json.loads(s3_obj["Body"].read().decode("utf-8"))

        #Filter resumes by job description ID

        if type == "resumes":
            filtered = {
                resume_id: data
                for resume_id, data in all_info.items()
                    if data.get("role") == job_description_id
            }

        elif type == "job description":
            filtered = all_info[job_description_id]
        
        return filtered
    
    except s3.exceptions.NoSuchKey:
        print("No resume file found.")
        return {}
    
    except Exception as e:
        print(f"Error loading resumes: {e}")
        return {}

def compute_gap_boost_score(job_chunk, resume_chunks, required_sections, similarity_threshold = 0, max_boost = 0.25):
    """
    Compute a boost score by comparing non-required resume sections to job-resume embedding gap.

        Args:
            job_chunk (dict): {section: {text, embedding}}
            resume_chunks (dict): {section: {text. embedding}}
            required_sections (tuple): Resume sections considered as core
            similarity_threshold (float): Minimum similarity to count as meaningful
            max_boost (float): Maximum value boost can contribute

        Returns:
            dict with:
                boost: float
                aligned_sections: list of resume section names contributing
                gap_vector: np.array (for debug)
    """

    job_vecs = [np.array(c["embedding"]) for c in job_chunk.values() if "embedding" in c]
    req_vecs = [np.array(resume_chunks[s]["embedding"]) for s in required_sections if s in resume_chunks]

    if not job_vecs or not req_vecs:
        return {"boost": 0.0,
                "aligned_sections": [],
                "gap_vector": None}
    
    job_mean = np.mean(job_vecs, axis = 0)
    req_mean = np.mean(req_vecs, axis = 0)
    gap_vec = job_mean - req_mean
    gap_vec = gap_vec.reshape(1, -1)

    boost_sections = []
    sim_scores = []

    for sec, data in resume_chunks.items():
        if sec in required_sections or "embedding" not in data:
            continue
        res_vec = np.array(data["embedding"]).reshape(1, -1)
        sim = cosine_similarity(gap_vec, res_vec)[0][0]

        if sim > similarity_threshold:
            boost_sections.append(sec)
            sim_scores.append(sim)

    if sim_scores:
        avg_sim = np.mean(sim_scores)
        boost = min(max_boost, avg_sim * max_boost)

    else:
        boost = 0.0

    return {
        "boost": round(boost, 3),
        "aligned_sections": boost_sections,
        "gap_vector": gap_vec
    }

def compute_resume_score_with_required_sections(
        section_scores,
        job_chunk, 
        resume_chunks,
        required_sections = ("skills", "experience", "education_studies"),
        optional_boost = True
):
    """
    Hybrid score combining required-section emphasis and optional boosts.

    Args:
        section_scores (dict): {section_name: similarity_score}
        required_sections (tuple): Key sections needed for a complete match
        optional_boost (bool): Whether to let other sections boost the score

    Returns:
        float: Final score between 0.0 and 1.0
    """

    required_scores = [section_scores[sec] for sec in required_sections if sec in section_scores]
    optional_scores = [score for sec, score in section_scores.items() if sec not in required_scores]

    #Compute base average from required sections
    if required_scores:
        base_score = np.mean(required_scores)

    else:
        base_score = 0.0

    # Optional section boost (e.g. Projects, Certifications)
    boost = 0.0

    if optional_boost and optional_scores:
        dict_boost = compute_gap_boost_score(job_chunk, resume_chunks, required_sections)
        boost = dict_boost["boost"]

    #Final score: average of required + optional boost

    final_score = base_score * (1 + boost)
    
    return final_score,dict_boost

def get_job_chunks(job_description_id):
    try:
        # Load the full resume database from S3
        s3_obj = s3.get_object(Bucket = S3_BUCKET,
                                      Key = "job_descriptions/job_descriptions.json")
        
        all_job_descriptions = json.loads(s3_obj["Body"].read().decode("utf-8"))

        job_data = all_job_descriptions[job_description_id]

        return job_data.get("chunks", {})

    except Exception as e:
        print("No job description with the id '{job_description_id}' was found")
        return {}
    
def get_matching_resumes(job_description_id):
    try:
        # Load the full resume database from S3
        s3_obj = s3.get_object(Bucket = S3_BUCKET,
                                      Key = "resumes/resumes.json")
        
        all_resumes = json.loads(s3_obj["Body"].read().decode("utf-8"))

        dict_filtered_resumes = {
            resume_id: data
                for resume_id, data in all_resumes.items()
                    if data.get("role") == job_description_id
        }

        return dict_filtered_resumes
    except Exception as e:
        print("No resumes for the job desription with the id '{job_descripton_id}' were found")
        return {}


def rank_resumes_for_job(job_chunks, dict_matching_resumes, top_n=5):
    """
    Compare all resumes linked to a specific job description and return the top N ranked matches.

    Args:
        job_description_id (str): UUID of the selected job description.
        job_descriptions_dict (dict): All job descriptions loaded from S3.
        resumes_dict (dict): All resumes loaded from S3.
        top_n (int): Number of top matches to return.

    Returns:
        list of dicts: Each with resume_id, score, file name, and matched sections.
    """

    job_chunks.pop('Average Annual Salary Range in Mexico (USD)')
    job_chunks.pop('Equal Opportunity Statement')
    job_chunks.pop('Application Process')


    job_vectors = [
        np.array(chunk["embedding"]).reshape(1, -1)
        for chunk in job_chunks.values()
        if "embedding" in chunk
    ]

    if not job_vectors:
        raise ValueError("Job description has no valid embeddings.")

    job_matrix = np.vstack(job_vectors)

    # 2. Filter resumes for the specific job

    results = []

    # 3. Compare each resume
    for resume_id, resume_data in dict_matching_resumes.items():
        resume_chunks = resume_data.get("chunks", {})
        resume_chunks.pop('personal_information')
        section_scores = {}
        sim_scores = []

        for res_sec, rc in resume_chunks.items():
            
            if "embedding" not in rc:
                continue
            vec = np.array(rc["embedding"]).reshape(1, -1)
            sim = cosine_similarity(vec, job_matrix)
            best_sim = float(np.max(sim))
            sim_scores.append(best_sim)
            section_scores[res_sec] = round(best_sim, 3)

        avg_score, dict_boost = compute_resume_score_with_required_sections(section_scores, job_chunks, resume_chunks)

        results.append({
            "resume_id": resume_id,
            "file": resume_data.get("file"),
            "score": avg_score,
            "section_scores": section_scores,
            "boost": dict_boost
        })

    # 4. Sort by score and return top N
    return sorted(results, key=lambda r: r["score"], reverse=True)[:top_n]

def compare_job_resume_embeddings(job_chunks: dict, resume_chunks: dict):
    """
    Compare embeddings between job description sections and resume sections.

    Args:
        job_chunks (dict): {"section_name": {"text": ..., "embedding": [...]}, ...}
        resume_chunks (dict): same structure

    Returns:
        dict: {
            "score": overall_score (float),
            "section_matches": {
                "Job Description": {
                    "best_resume_section": "general_information",
                    "similarity": 0.89
                },
                ...
            }
        }
    """
    results = []
    section_matches = {}

    for job_sec, job_data in job_chunks.items():
        job_vec = np.array(job_data["embedding"]).reshape(1, -1)
        best_score = 0
        best_match = None

        for res_sec, res_data in resume_chunks.items():
            res_vec = np.array(res_data["embedding"]).reshape(1, -1)
            sim = cosine_similarity(job_vec, res_vec)[0][0]

            if sim > best_score:
                best_score = sim
                best_match = res_sec

        section_matches[job_sec] = {
            "best_resume_section": best_match,
            "similarity": round(best_score, 3)
        }
        results.append(best_score)

    overall_score = round(float(np.mean(results)), 3) if results else 0.0

    return {
        "score": overall_score,
        "section_matches": section_matches
    }


def plot_job_based_radar_multi(match_results_dict, labels_by="Resume"):
    """
    Plot a radar chart comparing multiple resumes' match scores per job section.

    Args:
        match_results_dict (dict): {
            "resume_name_or_id": {
                "section_matches": {
                    "Job Description": {"similarity": 0.87, "best_resume_section": "..."},
                    ...
                }
            },
            ...
        }
        labels_by (str): 'Resume' or 'Job Section' for label style
    """
    # Extract all job sections from first entry
    first_key = next(iter(match_results_dict))
    job_sections = list(match_results_dict[first_key]["section_matches"].keys())

    # Complete the loop
    labels = job_sections
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    colors = ["#2196F3", "#4CAF50", "#FF5722"]  # Add more if needed
    for i, (resume_id, result) in enumerate(match_results_dict.items()):
        matches = result["section_matches"]
        scores = [matches[sec]["similarity"] for sec in job_sections]
        labels += [labels[0]]
        angles += [angles[0]]
        scores.append(scores[0])  # close the loop

        ax.plot(angles, scores, label=resume_id, linewidth=2, color=colors[i % len(colors)])
        ax.fill(angles, scores, alpha=0.15, color=colors[i % len(colors)])

    ax.set_title("Similarity by Job Description Section", size=14, pad=20)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), title="Resumes")

    st.pyplot(fig)



# ==== BACKEND: Test Generation Placeholder ====
def generate_test_from_description(description):
    prompt = f"Create a test with no more than 20 questions for this job description: {description} Include both technical and soft skills questions based on the context."


    body = json.dumps({
        "inferenceConfig": {
        "max_new_tokens": 1000
        },
        "messages": [
        {
            "role": "user",
            "content": [
                {
                    "text": prompt
                }
            ]
        }
        ]
    })
    try:
        response = bedrock.invoke_model(
            modelId= MODEL_ID,
            body=body
          )
        response_body = json.loads(response.get('body').read().decode('utf-8'))
        return response_body["output"]["message"]["content"][0]["text"]
    
    except Exception as e:
        return f"Error generating test: {e}"

# ==== STREAMLIT INTERFACE ====
st.title("AI-Powered Recruitment Assistant MVP")

# --- Job Description Generator ---
st.header("1. Generate Inclusive Job Description")
role_input = st.text_input("Enter Role Title", "Data Scientist")
region_input = st.selectbox(
     'Where will the employee be based?',
     ('Mexico','United States', 'South America'))
language_input = st.selectbox(
    'In what language should de Job Description be?',
    ('english', 'spanish')
)

# Generate and store job description
if st.button("Generate Description"):
    with st.spinner("Generating..."):
        full_output = generate_job_description(role_input, region_input, language_input)
        validation = validate_job_description(full_output, language_input)
        st.session_state["edited_description"] = full_output
        st.session_state["sections"] = split_job_description_sections(full_output)
        st.session_state["validation"] = validation

if "sections" in st.session_state:
    for section, content in st.session_state["sections"].items():
        updated = st.text_area(f"✏️ {section}", value=content, height=200)
        st.session_state["sections"][section] = updated  # update state

if "validation" in st.session_state:
    v = st.session_state["validation"]
    st.markdown("### ✅ Bias & Inclusion Review")
    st.markdown(f"**Result:** `{v.get('result', 'Unknown')}`")
    st.markdown(f"**Reasoning:** {v.get('reasoning', 'No reasoning provided.')}")
    
    if v.get("main_issues_identified"):
        st.markdown("**⚠️ Main Issues Identified:**")
        for issue in v["main_issues_identified"]:
            st.markdown(f"- {issue}")

    if v.get("recommendations"):
        st.markdown("**💡 Recommendations:**")
        for rec in v["recommendations"]:
            st.markdown(f"- {rec}")

if st.button("Save Job Description"):
    combined = ""
    sections = st.session_state["sections"]
    chunks = get_section_embeddings_dict(sections)

    for section, content in st.session_state["sections"].items():
        combined += f"**{section}**\n\n{content.strip()}\n\n"

    st.session_state["edited_description"] = combined.strip()
    
    save_message = save_job_description_to_json(
        role_input, region_input, language_input,
        chunks=chunks
    )
    st.success(save_message)

# --- Resume Upload ---
st.header("2. Upload Resume(s) for Matching")

job_map = load_job_descriptions_list()
if job_map:
    job_options = list(job_map.keys())
    job_labels = [job_map[jid]["label"] for jid in job_options]
    selected_job_id = st.selectbox("Select Job Description", options=job_options, format_func=lambda jid: job_map[jid]["label"])
    selected_job = job_map[selected_job_id]["data"]
else:
    selected_job_id = None
    selected_job = None
    st.warning("⚠️ No job descriptions available. Please create and save one first.")

uploaded_files = st.file_uploader("Upload PDF/Docx resume", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        file_name = str(uuid.uuid4())
        file_bytes = file.read()

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            tmp.seek(0)
            upload_message = upload_to_s3(tmp, f"resumes/files/{file_name}.pdf")
            st.success(upload_message)

        preview, parsed_text = parse_resume_to_text(file_bytes)
        
        if not parsed_text or len(parsed_text.strip()) < 50:
            st.warning("⚠️ Could not extract meaningful text from the uploaded resume.")
            continue

        lang = detect_resume_language(parsed_text)
        st.markdown(f"**Detected language:** `{lang}`")

        with st.spinner("Extracting sections..."):
            extracted = extract_resume_sections(parsed_text)

        if "error" in extracted:
            st.error(f"Extraction failed: {extracted['error']}")
        else:
            st.session_state["resume_sections"] = extracted

if "resume_sections" in st.session_state:
    for section, content in st.session_state["resume_sections"].items():
        formatted = format_section_for_display(content)
        updated = st.text_area(f"✏️ {section.title().replace('_', ' ')}", value=formatted, height=200)
        st.session_state["resume_sections"][section] = updated

if st.button("Save Resume"):
    language = detect_resume_language(parsed_text)
    file_id = file_name

    sections = st.session_state["resume_sections"]

    with st.spinner("Embedding sections..."):
        chunks_dict = embed_resume_sections(sections, bedrock)

    save_message = save_resume_to_json(selected_job_id,language,chunks_dict, file_id)
    st.success(save_message)
    
# --- Resume Vs Job Description Match ---
st.header("3. Job Descriptions matching")
job_matching_map = load_job_descriptions_list()
if job_matching_map:
    job_matching_options = list(job_matching_map.keys())
    job_matching_labels = [job_matching_map[jid]["label"] for jid in job_matching_options]
    job_matching_id = st.selectbox(
        "Select Job Description", 
        options=job_matching_options, 
        format_func=lambda jid: job_matching_map[jid]["label"],
        key = "match_selectbox"
    )
    job_matching = job_matching_map[job_matching_id]["data"]
    job_chunks = get_job_chunks(job_matching_id)
    dict_filtered_resumes = get_matching_resumes(job_matching_id)

if st.button("Match"):
    ranking_resumes = rank_resumes_for_job(job_chunks, dict_filtered_resumes, top_n=2)
    dict_match_results = {}

    for resume in ranking_resumes:
        ranked_resume_id = resume["resume_id"]
        resume_chunks = dict_filtered_resumes[ranked_resume_id]["chunks"]
        dict_match_results[resume["resume_id"]] = compare_job_resume_embeddings(job_chunks, resume_chunks)

    plot_job_based_radar_multi(dict_match_results)

else:
    job_matching_id = None
    job_matching = None
    #st.warning("⚠️ No job descriptions available. Please create and save one first.")

# --- Test Generator ---
st.header("4. Assessment Generator")
job_desc = st.text_area("Paste job description to generate a test")
if st.button("Generate Assessment"):
    with st.spinner("Generating test..."):
        test_output = generate_test_from_description(job_desc)
        st.text_area("Generated Assessment", test_output, height=400)
