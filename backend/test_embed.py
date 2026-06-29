"""Quick diagnostic: test Gemini embedding API and print the full response."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GEMINI_API_KEY", "")

print(f"Key loaded : {'YES' if key else 'NO - not in .env!'}")
print(f"Key length : {len(key)}")
print(f"Key prefix : {key[:6]}...")
print()

models = ["embedding-001", "text-embedding-004"]
versions = ["v1beta", "v1"]

for version in versions:
    for model in models:
        url = f"https://generativelanguage.googleapis.com/{version}/models/{model}:embedContent"
        print(f"Testing {version}/{model}...")
        r = requests.post(
            url,
            json={"content": {"parts": [{"text": "test"}]}},
            headers={"x-goog-api-key": key},
            timeout=15,
        )
        print(f"  Status : {r.status_code}")
        print(f"  Body   : {r.text[:300]}")
        print()
