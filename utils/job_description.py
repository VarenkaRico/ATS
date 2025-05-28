import re
import json
import uuid
from datetime import datetime
import os
from io import BytesIO


import boto3
from botocore.exceptions import NoCredentialsError

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
