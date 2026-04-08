"""
Predict route — POST /predict
Requires Authorization: Bearer <token> header.
Decodes base64 canvas image, runs the ML pipeline (unchanged), returns results.
"""

import sys
import os
import base64
import io
import traceback

from flask import Blueprint, request, jsonify
from PIL import Image

# ── Import unchanged ML utils from project root ───────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.preprocess import preprocess_image       # ← unchanged
from utils.predictor import UIComponentPredictor    # ← unchanged
from utils.html_mapper import map_class_to_html     # ← unchanged

predict_bp = Blueprint("predict", __name__)

# ── Load model once at startup ────────────────────────────────────────────────
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "ui_model.keras")

try:
    _predictor = UIComponentPredictor(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    _predictor = None

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_PAYLOAD_BYTES = 5 * 1024 * 1024   # 5 MB limit for base64 payload


def _validate_token(token: str) -> bool:
    """
    Validate the demo token (base64 of email).
    For production, replace with JWT verification.
    """
    from db import User
    try:
        email = base64.b64decode(token.encode()).decode()
        return User.query.filter_by(email=email).first() is not None
    except Exception:
        return False


def _react_code(label: str, html: str) -> str:
    comp = label.replace(" ", "").replace("_", "").capitalize()
    safe_html = html.replace("`", "'").replace("\n", " ")
    return f"""import React from 'react';

// Auto-generated React component for: {label}
const {comp} = () => {{
  return (
    <div style={{{{ fontFamily: 'Inter, sans-serif', padding: '16px' }}}}>
      {{/* {label} component */}}
      <div dangerouslySetInnerHTML={{{{ __html: `{safe_html}` }}}} />
    </div>
  );
}};

export default {comp};
"""


# ── POST /predict ─────────────────────────────────────────────────────────────

@predict_bp.route("/predict", methods=["POST"])
def predict():
    # 1. Auth guard — require Bearer token
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Authorization required. Please log in."}), 401

    token = auth_header[len("Bearer "):]
    if not _validate_token(token):
        return jsonify({"error": "Invalid or expired session. Please log in again."}), 401

    # 2. Parse payload
    data = request.get_json(force=True, silent=True) or {}
    image_data = data.get("image", "")

    if not image_data:
        return jsonify({"error": "No image data received."}), 400

    # 3. Size check — reject oversized payloads
    if len(image_data) > MAX_PAYLOAD_BYTES:
        return jsonify({"error": "Image payload too large (max 5 MB)."}), 413

    # 4. Decode base64 → PIL Image
    try:
        # Strip data-URL prefix if present (e.g. "data:image/png;base64,...")
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {str(e)}"}), 400

    # 5. ML prediction (completely unchanged logic) ────────────────────────────
    if _predictor is None:
        return jsonify({"error": "Model is not loaded. Please check server logs."}), 500

    try:
        processed = preprocess_image(img)                   # ← unchanged
        label, confidence = _predictor.predict(processed)  # ← unchanged
        html_code = map_class_to_html(label)                # ← unchanged
        react_code = _react_code(label, html_code)
        pct = int(float(confidence) * 100)

        return jsonify({
            "label":       label,
            "confidence":  pct,
            "html_code":   html_code,
            "react_code":  react_code,
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
