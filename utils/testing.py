import json
import boto3
import os

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