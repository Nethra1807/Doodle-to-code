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
import cv2
import numpy as np

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

def extract_components(img_pil: Image.Image) -> list:
    """
    Given a PIL image of the canvas, extract individual drawn components.
    Returns a list of cropped PIL images.
    """
    cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 10:
            boxes.append((x, y, w, h))
            
    if not boxes:
        width, height = img_pil.size
        return [(img_pil, (0, 0, width, height))]
        
    boxes = sorted(boxes, key=lambda b: b[1])
    
    cropped_data = []
    pad = 20
    height, width = thresh.shape
    
    for (x, y, w, h) in boxes:
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + w + pad)
        y2 = min(height, y + h + pad)
        
        crop = img_pil.crop((x1, y1, x2, y2))
        
        crop_w, crop_h = x2 - x1, y2 - y1
        max_dim = max(crop_w, crop_h)
        square_img = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        
        offset_x = (max_dim - crop_w) // 2
        offset_y = (max_dim - crop_h) // 2
        square_img.paste(crop, (offset_x, offset_y))
        
        cropped_data.append((square_img, (x, y, w, h)))
        
    return cropped_data


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
        components_data = extract_components(img)
        labels = []
        confidences = []
        html_codes = []
        
        CANVAS_W, CANVAS_H = 600, 450
        is_multi = len(components_data) > 1
        
        for comp_img, (x, y, w, h) in components_data:
            processed = preprocess_image(comp_img)
            label, confidence = _predictor.predict(processed)
            labels.append(label)
            confidences.append(float(confidence))
            
            raw_html = map_class_to_html(label)
            
            if is_multi:
                left_pct = (x / CANVAS_W) * 100
                top_pct = (y / CANVAS_H) * 100
                positioned_html = f'<div style="position: absolute; left: {left_pct:.2f}%; top: {top_pct:.2f}%;">{raw_html}</div>'
                html_codes.append(positioned_html)
            else:
                html_codes.append(raw_html)
            
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        final_label = "Multiple Components" if is_multi else labels[0]
        
        if is_multi:
            combined_html = f'<div style="position: relative; width: 100%; min-height: 450px; background: transparent;">\n'
            for h in html_codes:
                combined_html += f"  {h}\n"
            combined_html += '</div>'
        else:
            combined_html = html_codes[0]
            
        react_code = _react_code(final_label, combined_html)
        pct = int(avg_conf * 100)

        return jsonify({
            "label":       final_label,
            "confidence":  pct,
            "html_code":   combined_html,
            "react_code":  react_code,
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
