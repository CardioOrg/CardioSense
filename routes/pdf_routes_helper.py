"""
Shared PDF route helpers — build PDF responses and reusable data-fetching logic.
Eliminates duplication between patient_routes and doctor_routes.
"""

import json
import logging

from flask import Response

from services import pdf_service
from services.patient_service import get_profile, get_latest_component_run
from services.doctor_service import get_prescription
from services.firebase_service import get_db
from services.component4_chat_service import (
    get_patient_chat_sessions, get_session_messages
)

logger = logging.getLogger(__name__)


def pdf_response(pdf_bytes, filename, inline=False):
    """Build a Flask Response for a PDF download or inline view."""
    disposition = "inline" if inline else "attachment"
    return Response(
        bytes(pdf_bytes),
        mimetype="application/pdf",
        headers={
            "Content-Disposition": f"{disposition}; filename={filename}"
        }
    )


def parse_output_json(raw):
    """Safely parse output_json which may be a string or dict."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return raw if isinstance(raw, dict) else {}


def inject_chat_messages(output, uid):
    """For Component 4 results, inject chat messages if not already present."""
    if not output.get("messages"):
        sessions = get_patient_chat_sessions(uid)
        all_messages = []
        for s in sessions:
            msgs = get_session_messages(s.get("session_id", ""))
            all_messages.extend(msgs)
        all_messages.sort(key=lambda x: x.get("created_at", ""))
        output["messages"] = all_messages
    return output


def build_component_pdf(uid, component_id, fallback_redirect=None):
    """
    Build a component PDF for a given patient.  Returns (pdf_bytes, profile)
    or (None, None) if no results found.
    """
    profile = get_profile(uid) or {}
    run = get_latest_component_run(uid, component_id)
    if not run:
        return None, profile

    output = parse_output_json(run.get("output_json", {}))

    if component_id == 4:
        output = inject_chat_messages(output, uid)

    summary = run.get("summary_text", "")
    pdf_bytes = pdf_service.generate_component_pdf(
        profile, component_id, output, summary
    )
    return pdf_bytes, profile


def build_combined_pdf(uid):
    """
    Build a combined PDF for a given patient.  Returns (pdf_bytes, profile)
    or (None, None) if no results exist.
    """
    profile = get_profile(uid) or {}
    all_results = {}
    for cid in [1, 2, 3, 4]:
        run = get_latest_component_run(uid, cid)
        if run:
            run["output_json"] = parse_output_json(run.get("output_json", {}))
            all_results[cid] = run

    if not all_results:
        return None, profile

    pdf_bytes = pdf_service.generate_combined_pdf(profile, all_results)
    return pdf_bytes, profile


def build_prescription_pdf(prescription_id, requesting_uid=None):
    """
    Build a prescription PDF.  Returns (pdf_bytes, rx) or (None, None)
    if not found or access denied.
    """
    rx = get_prescription(prescription_id)
    if not rx:
        return None, None

    # If requesting_uid provided, check ownership
    if requesting_uid and rx.get("uid") != requesting_uid:
        return None, None

    patient_profile = get_profile(rx.get("uid", "")) or {}

    doctor_info = {}
    db = get_db()
    if db and rx.get("doctor_uid"):
        doc = db.collection("users").document(rx["doctor_uid"]).get()
        if doc.exists:
            doctor_info = doc.to_dict()

    pdf_bytes = pdf_service.generate_prescription_pdf(rx, patient_profile, doctor_info)
    return pdf_bytes, rx
