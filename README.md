# 🤖 AI-Powered Recruitment Assistant

A modular Streamlit-based MVP that leverages AWS Bedrock, S3, and semantic embeddings to automate and improve the recruitment process with fairness, transparency, and accuracy.

---

## 🚀 Features

- **Inclusive Job Description Generator**  
  Generates unbiased, compliant job descriptions tailored by role and region.

- **Bias Validation Agent**  
  Evaluates job descriptions for gendered, exclusionary, or unrealistic criteria.

- **Resume Section Extraction**  
  Automatically parses resumes into structured, editable sections.

- **Semantic Matching**  
  Uses Titan embeddings to compute cosine similarity between resumes and job descriptions.

- **Gap-Boost Scoring**  
  Detects where supplemental documents (e.g., cover letters) compensate for resume mismatches.

- **Custom Skill Assessment Generator**  
  Creates targeted assessments based on job criteria.

---

## 🧱 Tech Stack

- **Python** · Streamlit · AWS Bedrock (Titan Embeddings + Nova LLM)  
- **S3** for JSON persistence · **boto3** for AWS integration  
- `python-dotenv` for secure environment management  

---

## 🛠️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/VarenkaRico/ATS.git
cd ATS
```
### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure your environment
Create a .env file in the root directory with:

AWS_PROFILE=your-aws-cli-profile
S3_BUCKET=your-s3-bucket-name

### 4. Run the application
```bash
streamlit run Recruitment_app.py
```

## 📁 Project Structure

.
├── Recruitment_app.py         # Streamlit UI and logic
├── requirements.txt
├── .env                       # Local AWS credentials & config (not shared)
├── .env.example               # Template for collaborators
└── README.md

## ✅ Environment Variables
Variable	Description
AWS_PROFILE	AWS CLI profile name with Bedrock/S3 access
S3_BUCKET	S3 bucket name for storing data

The S3_BUCKET should be structured as follows
.
├── job_descriptions         # The json file for job descritptions will be stored here
├── resumes                  # The json file for resumes will be stored here
    └── files                # The original files uploaded will be stored here

## 🧠 Agentic Design Philosophy
Each function in this project acts as a specialized agent, with a chain-of-thought flow that includes:

Structured prompting
JSON parsing
Role-specific subagents
Score boosting via contextual augmentation
This enables composability, modular testing, and future scaling to LangChain, CrewAI, or multi-agent orchestration systems.

## 🔐 Security & Deployment Notes
Resume and job data are stored in S3 as structured JSON for auditability and compliance.
