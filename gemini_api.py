from flask import Flask, request, jsonify
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY", "").strip(' \'"'))

# ✅ USE THIS MODEL
model = genai.GenerativeModel("gemini-2.5-flash")

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt")

        response = model.generate_content(
            f"Generate clean HTML UI code only. No explanation.\n\n{prompt}"
        )

        # safer extraction
        output = ""
        if hasattr(response, "text") and response.text:
            output = response.text
        elif response.candidates:
            output = response.candidates[0].content.parts[0].text
        else:
            output = "No response generated"

        # Strip markdown code block formatting
        if output.startswith("```html"):
            output = output[7:]
        elif output.startswith("```"):
            output = output[3:]
            
        if output.endswith("```"):
            output = output[:-3]
            
        output = output.strip()

        return jsonify({"response": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5001, debug=True)