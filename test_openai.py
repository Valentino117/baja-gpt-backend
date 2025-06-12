# test_openai.py
import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("FINE_TUNED_MODEL_ID")

resp = openai.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "How are you?"}]
)
print(resp.choices[0].message.content.strip())
