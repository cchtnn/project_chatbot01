import requests
from config import get_settings

settings = get_settings()
key = settings.groq_api_key
model = settings.groq_model

url = "https://api.groq.com/openai/v1/chat/completions"
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
}
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

resp = requests.post(url, json=payload, headers=headers)
print(f"Status: {resp.status_code}")
print(f"Response: {resp.text}")
