"""
Doodle to Code - Flask Backend
Serves the HTML frontend and exposes ML prediction + auth endpoints.
ML code (utils/, model/) is completely unchanged.
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()
# ── Add project root to path so utils/ and model/ work unchanged ──────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, send_from_directory, send_file
from flask_cors import CORS

from db import db, User
import json

from routes.auth_routes import auth_bp
from routes.predict_routes import predict_bp
from routes.prompt_routes import prompt_bp
from routes.generate_routes import generate_bp

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(32)

# Configure SQLite DB
db_path = os.path.join(os.path.dirname(__file__), "users.db")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Enable CORS for all routes (allows frontend JS to call the API)
CORS(app, resources={r"/*": {"origins": "*"}})

db.init_app(app)

# ── Register blueprints ───────────────────────────────────────────────────────
app.register_blueprint(auth_bp)
app.register_blueprint(predict_bp)
app.register_blueprint(prompt_bp)
app.register_blueprint(generate_bp)

# ── Serve frontend static files ───────────────────────────────────────────────
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))


@app.route("/")
@app.route("/index.html")
def index():
    return send_file(os.path.join(FRONTEND_DIR, "index.html"))


@app.route("/login.html")
def login_page():
    return send_file(os.path.join(FRONTEND_DIR, "login.html"))


@app.route("/signup.html")
def signup_page():
    return send_file(os.path.join(FRONTEND_DIR, "signup.html"))


@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(os.path.join(FRONTEND_DIR, "css"), filename)


@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(os.path.join(FRONTEND_DIR, "js"), filename)


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(os.path.join(FRONTEND_DIR, "assets"), filename)


# ── Initial DB Migration & Setup ──────────────────────────────────────────────
def init_db():
    with app.app_context():
        db.create_all()
        # Migrate users from json if present and db is empty
        users_json = os.path.join(os.path.dirname(__file__), "users.json")
        if os.path.exists(users_json) and User.query.count() == 0:
            try:
                with open(users_json, "r") as f:
                    old_users = json.load(f)
                for ou in old_users:
                    new_u = User(
                        name=ou.get("name", "Unknown"),
                        email=ou.get("email", ""),
                        password=ou.get("password", ""),
                        phone=ou.get("phone", "")
                    )
                    db.session.add(new_u)
                db.session.commit()
                print("✅ Migrated users from users.json to SQLite database")
            except Exception as e:
                print(f"❌ Migration failed: {e}")

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print("\n" + "="*60)
    print("🎨 DOODLE TO CODE - SERVER RUNNING")
    print("="*60)
    print("\nClick the link below to open the app in your browser:")
    print("\n👉  http://localhost:5000  👈\n")
    print("="*60 + "\n")
    
    app.run(debug=False, host="0.0.0.0", port=5000)
