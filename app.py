# backend/app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import openai

# Load environment variables from .env
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL_ID")

print("🔑 Loaded OpenAI API key:", openai.api_key[:10], "...")
print("🎯 Fine-tuned model ID:", FINE_TUNED_MODEL)

app = Flask(__name__)

# ✅ CORS: Allow any origin (or tighten to your frontend's Vercel domain)
CORS(app, resources={r"/*": {"origins": "*"}})  # or set to https://your-vercel-url.vercel.app

@app.route("/", methods=["GET"])
def health_check():
    return "✅ Backend is running!"

@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.get_json()
        input_text = data.get("input", "")

        print("📥 Incoming input:", input_text)

        if not input_text:
            print("❌ Missing input text")
            return jsonify({"error": "Missing input"}), 400

        # Base GPT-4.1 with prompt to simulate Baja dialect
        print("🟦 Sending to base GPT-4.1...")
        vanilla_response = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "Translate the user's English input to Spanish using expressions common to Baja California. Keep it natural and local."
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ]
        )
        vanilla_output = vanilla_response.choices[0].message.content.strip()
        print("✅ Base model output:", vanilla_output)

        # Fine-tuned model without prompt
        print("🟩 Sending to fine-tuned GPT-4.1...")
        fine_tuned_response = openai.chat.completions.create(
            model=FINE_TUNED_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": input_text
                }
            ]
        )
        fine_tuned_output = fine_tuned_response.choices[0].message.content.strip()
        print("✅ Fine-tuned model output:", fine_tuned_output)

        return jsonify({
            "vanilla": vanilla_output,
            "fine_tuned": fine_tuned_output
        })

    except Exception as e:
        print("❌ Backend error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
