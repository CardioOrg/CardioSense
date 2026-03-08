"""
Cloud Storage service — upload / download helpers.
"""

import logging
import uuid
from datetime import timedelta

from services.firebase_service import get_bucket

logger = logging.getLogger(__name__)


def upload_file(file_obj, dest_folder, filename=None):
    """
    Upload a file object to Firebase Cloud Storage.
    Returns the storage path (not a public URL).
    """
    bucket = get_bucket()
    if not bucket:
        raise RuntimeError("Firebase Storage is not configured.")

    if not filename:
        ext = file_obj.filename.rsplit(".", 1)[-1] if "." in file_obj.filename else "bin"
        filename = f"{uuid.uuid4().hex}.{ext}"

    blob_path = f"{dest_folder}/{filename}"
    blob = bucket.blob(blob_path)
    blob.upload_from_file(file_obj, content_type=file_obj.content_type)
    logger.info("Uploaded file to %s", blob_path)
    return blob_path


def upload_bytes(data_bytes, dest_path, content_type="application/pdf"):
    """Upload raw bytes (e.g. a generated PDF) to Cloud Storage."""
    bucket = get_bucket()
    if not bucket:
        raise RuntimeError("Firebase Storage is not configured.")

    blob = bucket.blob(dest_path)
    blob.upload_from_string(data_bytes, content_type=content_type)
    logger.info("Uploaded bytes to %s", dest_path)
    return dest_path


def upload_bytes_public(data_bytes, dest_path, content_type="image/png"):
    """Upload raw bytes and make the blob publicly accessible.
    Returns the permanent public URL."""
    bucket = get_bucket()
    if not bucket:
        raise RuntimeError("Firebase Storage is not configured.")

    blob = bucket.blob(dest_path)
    blob.upload_from_string(data_bytes, content_type=content_type)
    blob.make_public()
    logger.info("Uploaded public blob to %s", dest_path)
    return blob.public_url


def get_download_url(storage_path, expiry_minutes=60):
    """Generate a signed download URL for a storage path."""
    bucket = get_bucket()
    if not bucket:
        return None

    blob = bucket.blob(storage_path)
    url = blob.generate_signed_url(expiration=timedelta(minutes=expiry_minutes))
    return url


def get_public_url(storage_path):
    """Make a blob publicly accessible and return its permanent public URL.
    Unlike signed URLs, these never expire."""
    bucket = get_bucket()
    if not bucket:
        return None

    blob = bucket.blob(storage_path)
    blob.make_public()
    return blob.public_url


def download_blob_bytes(storage_path):
    """Download blob as bytes."""
    bucket = get_bucket()
    if not bucket:
        return None
    blob = bucket.blob(storage_path)
    return blob.download_as_bytes()

