"""
CardioSense — Central configuration.
Reads from environment variables (see .env.example).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Directories ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ── Flask ──────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")

# ── Firebase ───────────────────────────────────────────────────────
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")
FIREBASE_SERVICE_ACCOUNT_JSON_PATH = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT_JSON_PATH", ""
)
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET", "")
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY", "")

# ── Groq / SLM ────────────────────────────────────────────────────
CARDIOSENSE_KEY = os.getenv("CARDIOSENSE_KEY", "")

# ── Model file paths (reference existing component folders) ────────
COMPONENT1_MODEL_PATH = BASE_DIR / "CardioSense_Component1" / "cardiosense_component1_model.pkl"
COMPONENT1_DATA_PATH = BASE_DIR / "CardioSense_Component1" / "structured_dataset.csv"

COMPONENT2_MODEL_PATH = BASE_DIR / "CardioSense_Component2" / "cardiosense_cxr_densenet121.pth"
COMPONENT2_LABEL_MAP_PATH = BASE_DIR / "CardioSense_Component2" / "label_map.json"

COMPONENT3_MODELS_DIR = BASE_DIR / "CardioSense_Component3" / "models"
COMPONENT3_META_PATH = BASE_DIR / "CardioSense_Component3" / "meta.json"

# ── Groq model for chatbot and summaries ───────────────────────────
GROQ_CHAT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── Image settings ─────────────────────────────────────────────────
CXR_IMG_SIZE = 224
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
