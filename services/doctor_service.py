"""
Doctor service — patient listing, booking management, prescription CRUD.
Doctors can read all patient data.
"""

import json
import logging
from datetime import datetime
from services.firebase_service import get_db

logger = logging.getLogger(__name__)


# ── Reusable Firestore query helper ──────────────────────────────

def _query_collection(db, collection_name, uid, sort_field="created_at",
                      descending=True):
    """
    Query a Firestore collection by uid, return list of dicts with 'id' set.
    Sorted by *sort_field* (descending by default).
    """
    results = []
    for doc in db.collection(collection_name).where("uid", "==", uid).stream():
        d = doc.to_dict()
        d["id"] = doc.id
        results.append(d)
    results.sort(key=lambda x: x.get(sort_field, ""), reverse=descending)
    return results


# ── Patient listing ────────────────────────────────────────────────

def get_all_patients():
    """Return all users with role=patient plus their profiles."""
    db = get_db()
    if not db:
        return []

    docs = db.collection("users").where("role", "==", "patient").stream()
    patients = []
    for doc in docs:
        user = doc.to_dict()
        uid = user.get("uid", doc.id)
        profile_doc = db.collection("patient_profiles").document(uid).get()
        profile = profile_doc.to_dict() if profile_doc.exists else {}
        user["profile"] = profile
        patients.append(user)
    logger.debug("Fetched %d patients", len(patients))
    return patients


def get_all_doctors():
    """Return all users with role=doctor."""
    db = get_db()
    if not db:
        return []
    docs = db.collection("users").where("role", "==", "doctor").stream()
    return [doc.to_dict() for doc in docs]


def get_patient_detail(uid):
    """Get full patient detail: profile, records, component runs, bookings,
    prescriptions, and chat sessions."""
    db = get_db()
    if not db:
        return {}

    profile_doc = db.collection("patient_profiles").document(uid).get()
    profile = profile_doc.to_dict() if profile_doc.exists else {}

    # User info
    user_doc = db.collection("users").document(uid).get()
    user_info = user_doc.to_dict() if user_doc.exists else {}

    # Use helper for consistent query + sort
    records = _query_collection(db, "patient_records", uid)
    component_runs = _query_collection(db, "component_runs", uid)
    bookings = _query_collection(db, "bookings", uid)
    prescriptions = _query_collection(db, "prescriptions", uid)

    # Chat sessions need nested messages — can't fully use helper
    chat_sessions = []
    for doc in db.collection("chat_sessions").where("uid", "==", uid).stream():
        sd = doc.to_dict()
        sd["id"] = doc.id
        msgs = []
        for mdoc in (db.collection("chat_messages")
                     .where("session_id", "==", doc.id)
                     .stream()):
            msgs.append(mdoc.to_dict())
        msgs.sort(key=lambda x: x.get("created_at", ""))
        sd["messages"] = msgs
        chat_sessions.append(sd)
    chat_sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    logger.debug("Fetched detail for patient uid=%s", uid)
    return {
        "user_info": user_info,
        "profile": profile,
        "records": records,
        "component_runs": component_runs,
        "bookings": bookings,
        "prescriptions": prescriptions,
        "chat_sessions": chat_sessions,
    }


# ── Booking management ────────────────────────────────────────────

def get_all_bookings():
    """Return all bookings for all patients."""
    db = get_db()
    if not db:
        return []
    docs = db.collection("bookings").stream()
    results = []
    for doc in docs:
        d = doc.to_dict()
        d["id"] = doc.id
        # Attach patient name
        uid = d.get("uid", "")
        if uid:
            user_doc = db.collection("users").document(uid).get()
            if user_doc.exists:
                d["patient_name"] = user_doc.to_dict().get("full_name", uid)
            else:
                d["patient_name"] = uid
        results.append(d)
    results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return results


def update_booking_status(booking_id, status, doctor_uid):
    db = get_db()
    if not db:
        return
    db.collection("bookings").document(booking_id).update({
        "status": status,
        "doctor_uid": doctor_uid,
        "updated_at": datetime.utcnow().isoformat(),
    })
    logger.info("Booking %s updated to '%s' by doctor %s",
                booking_id, status, doctor_uid)


# ── Prescriptions ──────────────────────────────────────────────────

def create_prescription(data):
    """Create a prescription document. Only callable by a doctor."""
    db = get_db()
    if not db:
        return None
    now = datetime.utcnow().isoformat()

    med_list = data.get("medication_list_json", "[]")
    if isinstance(med_list, str):
        try:
            json.loads(med_list)
        except json.JSONDecodeError:
            med_list = json.dumps([{"medication": med_list}])

    prescription = {
        "uid": data.get("uid", ""),
        "doctor_uid": data.get("doctor_uid", ""),
        "booking_id": data.get("booking_id", ""),
        "medication_list_json": med_list,
        "instructions_text": data.get("instructions_text", ""),
        "followup_text": data.get("followup_text", ""),
        "pdf_storage_path": data.get("pdf_storage_path", ""),
        "created_at": now,
    }
    ref = db.collection("prescriptions").add(prescription)
    prescription["prescription_id"] = ref[1].id
    db.collection("prescriptions").document(ref[1].id).update(
        {"prescription_id": ref[1].id}
    )
    logger.info("Prescription %s created for patient %s",
                ref[1].id, data.get("uid", ""))
    return prescription


def get_prescription(prescription_id):
    db = get_db()
    if not db:
        return None
    doc = db.collection("prescriptions").document(prescription_id).get()
    if doc.exists:
        d = doc.to_dict()
        d["id"] = doc.id
        return d
    return None

