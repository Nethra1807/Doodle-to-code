from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import os
import requests  # ✅ ADDED

from dotenv import load_dotenv

from utils.preprocess import preprocess_image
from utils.predictor import UIComponentPredictor
from utils.html_mapper import map_class_to_html

# =========================
# 🔹 ENV SETUP
# =========================
load_dotenv()

# =========================
# 🔹 FLASK SETUP
# =========================
app = Flask(__name__)
CORS(app)

# =========================
# 🔹 LOAD MODEL SAFELY
# =========================
MODEL_PATH = "model/ui_model.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

predictor = UIComponentPredictor(MODEL_PATH)

# =========================
# 🔹 ROUTE: HEALTH CHECK
# =========================
@app.route("/", methods=["GET"])
def home():
    return "Backend is running 🚀"

# =========================
# 🔹 ROUTE: ML PREDICT
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        image_data = data["image"]

        if "," not in image_data:
            return jsonify({"error": "Invalid image format"}), 400

        image_bytes = base64.b64decode(image_data.split(",")[1])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        processed = preprocess_image(image)
        label, confidence = predictor.predict(processed)
        html = map_class_to_html(label)

        return jsonify({
            "label": label,
            "confidence": float(confidence),
            "html": html
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# 🔹 ROUTE: GEMINI GENERATE (API VERSION)
# =========================
@app.route("/generate", methods=["POST"])
@app.route("/generate", methods=["POST"])
def generate_proxy():
    import requests

    try:
        data = request.json

        response = requests.post(
            "http://127.0.0.1:5001/generate",
            json=data
        )

        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# 🔹 AUTH: SIGNUP
# =========================
@app.route("/signup", methods=["POST", "OPTIONS"])
def signup():
    if request.method == "OPTIONS":
        return "", 200

    data = request.json

    return jsonify({
        "message": "Signup successful (dummy)",
        "user": data
    })


# =========================
# 🔹 AUTH: LOGIN
# =========================
@app.route("/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return "", 200

    data = request.json

    return jsonify({
        "message": "Login successful (dummy)",
        "token": "fake-jwt-token",
        "user": data
    })


# =========================
# 🔹 RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)