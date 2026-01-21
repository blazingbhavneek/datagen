import logging
import http.client
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import requests
import json

# Enable HTTP debug logging
http.client.HTTPConnection.debuglevel = 1
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.DEBUG)

# Define Pydantic model for structured output
class ResponseModel(BaseModel):
    answer: str
    confidence: float

# Setup vLLM client (OpenAI compatible API)
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="qwen",
    temperature=0.7,
    max_tokens=512
)

# Create structured output chain
structured_llm = llm.with_structured_output(ResponseModel)

# Make request using LangChain
print("=== LangChain Request ===")
result = structured_llm.invoke("What is the capital of France?")
print(result)

# Now replicate the exact same request using requests library
print("\n=== Requests Library Request (Replicating LangChain) ===")
url = "http://localhost:8000/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "X-Stainless-Helper-Method": "chat.completions.parse",
    "X-Stainless-Raw-Response": "true"
}

payload = {
    "messages": [
        {
            "content": "What is the capital of France?",
            "role": "user"
        }
    ],
    "model": "qwen",
    "max_completion_tokens": 512,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "properties": {
                    "answer": {"title": "Answer", "type": "string"},
                    "confidence": {"title": "Confidence", "type": "number"}
                },
                "required": ["answer", "confidence"],
                "title": "ResponseModel",
                "type": "object",
                "additionalProperties": False
            },
            "name": "ResponseModel",
            "strict": True
        }
    },
    "stream": False,
    "temperature": 0.7
}

# Send the request
response = requests.post(url, headers=headers, json=payload)

# Print the response
print(f"Status Code: {response.status_code}")
print("Response Headers:")
for key, value in response.headers.items():
    print(f"  {key}: {value}")
print("\nResponse Body:")
print(json.dumps(response.json(), indent=2))

# Parse the structured output from response
if response.status_code == 200:
    response_data = response.json()
    choices = response_data.get('choices', [])
    if choices and len(choices) > 0:
        message_content = choices[0].get('message', {}).get('content', '')
        try:
            # The content should be JSON since we requested structured output
            structured_result = json.loads(message_content)
            print("\nParsed Structured Result:")
            print(f"answer='{structured_result['answer']}' confidence={structured_result['confidence']}")
        except json.JSONDecodeError:
            print("\nError parsing JSON from response content")
            print(f"Raw content: {message_content}")
