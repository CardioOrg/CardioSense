"""
Patient service — profile, records, component runs, bookings, prescriptions.
All operations are scoped to a single patient UID.
"""

import json
import logging
from datetime import datetime

from services.firebase_service import get_db

logger = logging.getLogger(__name__)


# ── Patient Profile ────────────────────────────────────────────────

def get_profile(uid):
    db = get_db()
    if not db:
        return None
    doc = db.collection("patient_profiles").document(uid).get()
    return doc.to_dict() if doc.exists else None


def save_profile(uid, data):
    db = get_db()
    if not db:
        return
    now = datetime.utcnow().isoformat()

    # Auto-compute BMI
    try:
        h = float(data.get("height", 0))
        w = float(data.get("weight", 0))
        if h > 0:
            data["bmi"] = round(w / ((h / 100.0) ** 2), 1)
    except (ValueError, TypeError):
        pass

    profile = {
        "uid": uid,
        "full_name": data.get("full_name", ""),
        "dob": data.get("dob", ""),
        "sex": data.get("sex", ""),
        "phone": data.get("phone", ""),
        "address": data.get("address", ""),
        "emergency_contact": data.get("emergency_contact", ""),
        "medical_history_text": data.get("medical_history_text", ""),
        "medications_text": data.get("medications_text", ""),
        "allergies_text": data.get("allergies_text", ""),
        "risk_factors_text": data.get("risk_factors_text", ""),
        # Clinical fields needed for component inputs
        "age_years": _to_int(data.get("age_years", 0)),
        "height": _to_float(data.get("height", 0)),
        "weight": _to_float(data.get("weight", 0)),
        "bmi": _to_float(data.get("bmi", 0)),
        "ap_hi": _to_int(data.get("ap_hi", 120)),
        "ap_lo": _to_int(data.get("ap_lo", 80)),
        "cholesterol": _to_int(data.get("cholesterol", 1)),
        "gluc": _to_int(data.get("gluc", 1)),
        "smoke": _to_int(data.get("smoke", 0)),
        "alco": _to_int(data.get("alco", 0)),
        "active": _to_int(data.get("active", 0)),
        # UCI Advanced Clinical Data
        "cp": _to_int(data.get("cp", 1)),
        "fbs": _to_int(data.get("fbs", 0)),
        "restecg": _to_int(data.get("restecg", 0)),
        "thalach": _to_int(data.get("thalach", 150)),
        "exang": _to_int(data.get("exang", 0)),
        "oldpeak": _to_float(data.get("oldpeak", 1.0)),
        "slope": _to_int(data.get("slope", 1)),
        "ca": _to_int(data.get("ca", 1)),
        "thal": _to_int(data.get("thal", 2)),
        "updated_at": now,
    }

    existing = db.collection("patient_profiles").document(uid).get()
    if existing.exists:
        db.collection("patient_profiles").document(uid).update(profile)
    else:
        profile["created_at"] = now
        db.collection("patient_profiles").document(uid).set(profile)

    return profile


# ── Patient Records ────────────────────────────────────────────────

def add_record(uid, data):
    db = get_db()
    if not db:
        return None
    now = datetime.utcnow().isoformat()
    record = {
        "uid": uid,
        "record_type": data.get("record_type", "vitals"),
        "ap_hi": _to_int(data.get("ap_hi", 0)),
        "ap_lo": _to_int(data.get("ap_lo", 0)),
        "cholesterol": _to_int(data.get("cholesterol", 1)),
        "gluc": _to_int(data.get("gluc", 1)),
        "bmi": _to_float(data.get("bmi", 0)),
        "height": _to_float(data.get("height", 0)),
        "weight": _to_float(data.get("weight", 0)),
        "smoke": _to_int(data.get("smoke", 0)),
        "alco": _to_int(data.get("alco", 0)),
        "active": _to_int(data.get("active", 0)),
        "age_years": _to_int(data.get("age_years", 0)),
        # UCI Advanced Clinical Data
        "cp": _to_int(data.get("cp", 1)),
        "fbs": _to_int(data.get("fbs", 0)),
        "restecg": _to_int(data.get("restecg", 0)),
        "thalach": _to_int(data.get("thalach", 150)),
        "exang": _to_int(data.get("exang", 0)),
        "oldpeak": _to_float(data.get("oldpeak", 1.0)),
        "slope": _to_int(data.get("slope", 1)),
        "ca": _to_int(data.get("ca", 1)),
        "thal": _to_int(data.get("thal", 2)),
        "notes": data.get("notes", ""),
        "cxr_storage_path": data.get("cxr_storage_path", ""),
        "created_at": now,
    }
    ref = db.collection("patient_records").add(record)
    record["id"] = ref[1].id
    return record


def get_records(uid, record_type=None):
    db = get_db()
    if not db:
        return []
    query = db.collection("patient_records").where("uid", "==", uid)
    if record_type:
        query = query.where("record_type", "==", record_type)
    docs = query.stream()
    results = []
    for doc in docs:
        d = doc.to_dict()
        d["id"] = doc.id
        results.append(d)
    results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return results


def get_latest_record(uid, record_type=None):
    records = get_records(uid, record_type)
    return records[0] if records else None


# ── Component Runs ─────────────────────────────────────────────────

def save_component_run(uid, component_id, input_ref, output_json,
                       summary_text=""):
    db = get_db()
    if not db:
        return None
    now = datetime.utcnow().isoformat()
    run = {
        "uid": uid,
        "component_id": component_id,
        "input_ref": input_ref,
        "output_json": json.dumps(output_json) if isinstance(output_json, (dict, list)) else str(output_json),
        "summary_text": summary_text,
        "created_at": now,
    }
    ref = db.collection("component_runs").add(run)
    run["id"] = ref[1].id
    return run


def get_component_runs(uid, component_id=None):
    db = get_db()
    if not db:
        return []
    query = db.collection("component_runs").where("uid", "==", uid)
    if component_id is not None:
        query = query.where("component_id", "==", int(component_id))
    docs = query.stream()
    results = []
    for doc in docs:
        d = doc.to_dict()
        d["id"] = doc.id
        # Deserialize output_json if stored as string
        oj = d.get("output_json")
        if isinstance(oj, str):
            try:
                d["output_json"] = json.loads(oj)
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(d)
    results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return results


def get_latest_component_run(uid, component_id):
    runs = get_component_runs(uid, component_id)
    return runs[0] if runs else None


# ── Bookings ───────────────────────────────────────────────────────

def create_booking(uid, data):
    db = get_db()
    if not db:
        return None
    now = datetime.utcnow().isoformat()
    booking = {
        "uid": uid,
        "doctor_uid": data.get("doctor_uid", ""),
        "status": "requested",
        "preferred_date": data.get("preferred_date", ""),
        "preferred_time": data.get("preferred_time", ""),
        "reason": data.get("reason", ""),
        "created_at": now,
        "updated_at": now,
    }
    ref = db.collection("bookings").add(booking)
    booking["booking_id"] = ref[1].id
    # Also store the booking_id inside the document
    db.collection("bookings").document(ref[1].id).update(
        {"booking_id": ref[1].id}
    )
    return booking


def get_bookings(uid):
    db = get_db()
    if not db:
        return []
    docs = (db.collection("bookings")
            .where("uid", "==", uid)
            .stream())
    results = []
    for doc in docs:
        d = doc.to_dict()
        d["id"] = doc.id
        results.append(d)
    results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return results


# ── Prescriptions ──────────────────────────────────────────────────

def get_prescriptions(uid):
    db = get_db()
    if not db:
        return []
    docs = (db.collection("prescriptions")
            .where("uid", "==", uid)
            .stream())
    results = []
    for doc in docs:
        d = doc.to_dict()
        d["id"] = doc.id
        results.append(d)
    results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return results


# ── Helpers ────────────────────────────────────────────────────────

def _to_int(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _to_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0
