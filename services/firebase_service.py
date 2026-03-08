"""
Firebase Admin SDK initialisation — singleton pattern.
"""

import logging

import firebase_admin
from firebase_admin import credentials, firestore, storage
import config

logger = logging.getLogger(__name__)

_initialized = False
db = None
bucket = None


def init_firebase():
    """Initialise Firebase Admin SDK once. Safe to call multiple times."""
    global _initialized, db, bucket
    if _initialized:
        return

    if not config.FIREBASE_SERVICE_ACCOUNT_JSON_PATH:
        # Allow app to start without Firebase for template development
        logger.warning("No Firebase service account configured — running without Firebase.")
        _initialized = True
        return

    try:
        cred = credentials.Certificate(config.FIREBASE_SERVICE_ACCOUNT_JSON_PATH)
        firebase_admin.initialize_app(cred, {
            "storageBucket": config.FIREBASE_STORAGE_BUCKET,
        })

        db = firestore.client()
        bucket = storage.bucket()
        _initialized = True
        logger.info("Firebase Admin SDK initialised successfully.")
    except Exception:
        logger.exception("Failed to initialise Firebase Admin SDK.")
        _initialized = True  # Prevent repeated attempts


def get_db():
    """Return Firestore client (may be None if unconfigured)."""
    return db


def get_bucket():
    """Return Cloud Storage bucket (may be None if unconfigured)."""
    return bucket
