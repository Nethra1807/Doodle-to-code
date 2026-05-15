"""
Auth routes — /login and /signup
Passwords are hashed with bcrypt. Users stored in users.json.
"""

import base64
import bcrypt
from flask import Blueprint, request, jsonify
from db import db, User

auth_bp = Blueprint("auth", __name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_token(email: str) -> str:
    """Simple demo token: base64 of email. Replace with JWT for production."""
    return base64.b64encode(email.encode()).decode()


def _verify_token(token: str) -> bool:
    """Verify that the token corresponds to a registered user."""
    try:
        email = base64.b64decode(token.encode()).decode()
        return User.query.filter_by(email=email).first() is not None
    except Exception:
        return False


# ── /signup ───────────────────────────────────────────────────────────────────

@auth_bp.route("/signup", methods=["POST"])
def signup():
    data = request.get_json(force=True, silent=True) or {}

    name     = (data.get("name") or "").strip()
    email    = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "")
    confirm  = (data.get("confirm") or "")
    phone    = (data.get("phone") or "").strip()

    # Basic validation
    if not name:
        return jsonify({"error": "Full name is required."}), 400
    if not email or "@" not in email:
        return jsonify({"error": "A valid email is required."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400
    if password != confirm:
        return jsonify({"error": "Passwords do not match."}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "An account with this email already exists."}), 409

    # Hash password with bcrypt
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    new_user = User(name=name, email=email, password=hashed, phone=phone)
    db.session.add(new_user)
    db.session.commit()

    token = _make_token(email)
    return jsonify({"token": token, "name": name, "email": email}), 201


# ── /login ────────────────────────────────────────────────────────────────────

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True, silent=True) or {}

    email    = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "")

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    user = User.query.filter_by(email=email).first()

    if user is None:
        return jsonify({"error": "No account found with this email."}), 401

    # Verify with bcrypt
    if not bcrypt.checkpw(password.encode(), user.password.encode()):
        return jsonify({"error": "Incorrect password."}), 401

    token = _make_token(email)
    return jsonify({"token": token, "name": user.name, "email": email}), 200


# ── /verify-token (internal use by predict route) ────────────────────────────

@auth_bp.route("/verify-token", methods=["POST"])
def verify_token():
    data  = request.get_json(force=True, silent=True) or {}
    token = data.get("token", "")
    if _verify_token(token):
        return jsonify({"valid": True}), 200
    return jsonify({"valid": False}), 401
