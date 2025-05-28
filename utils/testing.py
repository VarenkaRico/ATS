import json
import boto3
import os

from utils.aws_clients import s3, bedrock, S3_BUCKET, MODEL_NOVA, MODEL_TITAN_EMBED

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
            modelId= MODEL_NOVA,
            body=body
          )
        response_body = json.loads(response.get('body').read().decode('utf-8'))
        return response_body["output"]["message"]["content"][0]["text"]
    
    except Exception as e:
        return f"Error generating test: {e}"