"""
Generate route — POST /generate
Accepts a JSON body: { "prompt": "..." }
Calls the OpenAI API and returns generated HTML + React code.
API key is read securely from the OPENAI_API_KEY environment variable.
"""

import os
import traceback

from dotenv import load_dotenv
from flask import Blueprint, request, jsonify

# Load .env file — works whether run from project root or backend/
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_env_path)

# Import shared helpers from predict_routes (same package, relative import safe)
from routes.predict_routes import _react_code

generate_bp = Blueprint("generate", __name__)


@generate_bp.route("/generate", methods=["POST"])
def generate():
    # ── 1. Parse request body ─────────────────────────────────────────────────
    data = request.get_json(force=True, silent=True) or {}
    user_prompt = data.get("prompt", "").strip()

    if not user_prompt:
        return jsonify({"error": "Prompt cannot be empty."}), 400

    # ── 2. Check for API key ──────────────────────────────────────────────────
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return jsonify({
            "error": "GEMINI_API_KEY is not set. Please add it to your backend/.env file."
        }), 500

    # ── 3. Call Gemini API ────────────────────────────────────────────────────
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        system_prompt = (
            "You are an expert Frontend Developer.\n"
            "Generate a clean, responsive UI component based on the user's description.\n\n"
            "Rules:\n"
            "- Use Tailwind CSS for styling (it will be loaded via CDN — do NOT add the CDN tag yourself).\n"
            "- Return ONLY raw HTML. No <html>, <head>, or <body> wrappers.\n"
            "- Do NOT wrap output in markdown code fences (no ```html).\n"
            "- Start with a single root <div> that has padding, e.g. <div class=\"p-8 ...\">\n"
            "- Make it modern, responsive, and visually attractive."
        )

        response = model.generate_content(
            f"{system_prompt}\n\nUser Request: {user_prompt}"
        )

        html_code = ""
        if hasattr(response, "text") and response.text:
            html_code = response.text.strip()
        elif response.candidates:
            html_code = response.candidates[0].content.parts[0].text.strip()

        # Strip any markdown fences the model might still add
        if html_code.startswith("```html"):
            html_code = html_code[7:]
        elif html_code.startswith("```"):
            html_code = html_code[3:]
        if html_code.endswith("```"):
            html_code = html_code[:-3]
        html_code = html_code.strip()

        # Inject Tailwind CDN so the iframe preview renders correctly
        tailwind_cdn = '<script src="https://cdn.tailwindcss.com"></script>'
        if tailwind_cdn not in html_code:
            html_code = tailwind_cdn + "\n" + html_code

        react_code = _react_code("GeneratedComponent", html_code)

        return jsonify({
            "html_code":   html_code,
            "react_code":  react_code,
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gemini API call failed: {str(e)}"}), 500
