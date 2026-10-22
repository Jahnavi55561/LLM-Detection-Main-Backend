

from flask import Flask, render_template, request, jsonify
from datetime import datetime, timezone
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import re
from flask_cors import CORS

from services.evidence import retrieve_evidence
from services.verification import verify_claim, model_status
from services.correction import (
    suggest_correction,
    validate_correction,
    build_corrected_paragraph,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("app")

app = Flask(__name__)

# ============================================================
# CORS CONFIGURATION
# Allow the deployed Vercel frontend to call this Render API.
# No cookies/credentials are used by this application.
# ============================================================

FRONTEND_ORIGIN = "https://llm-detection-main-frontend.vercel.app"

CORS(
    app,
    resources={
        r"/api/*": {
            # Keep "*" while debugging deployment CORS so Vercel
            # requests and browser OPTIONS preflight are not blocked.
            "origins": "*",
            "methods": ["GET", "POST", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "max_age": 600,
        }
    },
    supports_credentials=False,
    always_send=True,
    vary_header=True,
)

@app.after_request
def add_cors_headers(response):
    # Explicit fallback for API responses, including error responses.
    # This also helps make CORS behavior consistent across Render workers.
    if request.path.startswith("/api/"):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Max-Age"] = "600"
    return response


RECENT_HISTORY = deque(maxlen=20)

# ============================================================
# 30 CURATED CLAIMS: 15 SUPPORTED + 15 CONTRADICTED
# ============================================================

CURATED_FACTS = [
    {
        "claims": ["The capital of Australia is Canberra."],
        "status": "Supported",
        "evidence": "Canberra is the capital city of Australia.",
        "correction": "",
        "domain": "Geography",
    },
    {
        "claims": ["The Earth revolves around the Sun."],
        "status": "Supported",
        "evidence": "Earth orbits the Sun once approximately every 365.25 days.",
        "correction": "",
        "domain": "Astronomy",
    },
    {
        "claims": ["IPv4 addresses are 32 bits long."],
        "status": "Supported",
        "evidence": "An IPv4 address contains 32 bits, commonly written as four decimal octets.",
        "correction": "",
        "domain": "Computer Networks",
    },
    {
        "claims": ["The human heart has four chambers."],
        "status": "Supported",
        "evidence": "The human heart has four chambers: two atria and two ventricles.",
        "correction": "",
        "domain": "Biology",
