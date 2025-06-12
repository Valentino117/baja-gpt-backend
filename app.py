import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import openai

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL_ID")

if not openai.api_key:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")
if not FINE_TUNED_MODEL:
    raise RuntimeError("Missing FINE_TUNED_MODEL_ID in environment")

print("🔑 Loaded OpenAI API key:", openai.api_key[:10], "...")
print("🎯 Fine-tuned model ID:", FINE_TUNED_MODEL)

app = Flask(__name__)
# Allow any origin – lock this down in prod
CORS(app, resources={r"/*": {"origins": "*"}})

SYSTEM_PROMPT = (
    "Translate the user's English input to Spanish using expressions common to Baja California. "
    "Keep it natural, casual, and local."
)

@app.route("/", methods=["GET"])
def health_check():
    return "✅ Backend is running!"

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json(silent=True) or {}
    input_text = data.get("input", "").strip()
    print("📥 Incoming input:", input_text)

    if not input_text:
        return jsonify({"error": "Missing input"}), 400

    try:
        # Base (vanilla) model call
        print("🟦 Querying base GPT-4.1...")
        vanilla_resp = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": input_text}
            ],
        )
        vanilla_output = vanilla_resp.choices[0].message.content.strip()
        print("✅ Base model:", vanilla_output)

        # Fine-tuned model call with same prompt
        print("🟩 Querying fine-tuned model...")
        ft_resp = openai.chat.completions.create(
            model=FINE_TUNED_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": input_text}
            ],
        )
        fine_tuned_output = ft_resp.choices[0].message.content.strip()
        print("✅ Fine-tuned model:", fine_tuned_output)

        return jsonify({
            "vanilla": vanilla_output,
            "fine_tuned": fine_tuned_output
        })

    except Exception as e:
        print("❌ Error in backend:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
