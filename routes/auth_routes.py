"""
Auth routes — login, signup, session management.
"""

import logging
from datetime import datetime

from flask import (
    Blueprint, render_template, request, redirect, url_for, flash, session,
    jsonify
)
from firebase_admin import auth as fb_auth

from services.auth_service import (
    verify_id_token, create_session, clear_session, signup_user
)
from services.firebase_service import get_db

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login")
def login():
    return render_template("login.html")


@auth_bp.route("/signup/patient")
def signup_patient():
    return render_template("signup_patient.html")


@auth_bp.route("/signup/doctor")
def signup_doctor():
    return render_template("signup_doctor.html")


@auth_bp.route("/auth/session-login", methods=["POST"])
def session_login():
    """
    Receives Firebase ID token from frontend JS after client-side auth.
    Verifies the token and creates a Flask session.
    """
    data = request.get_json(silent=True) or {}
    id_token = data.get("idToken", "")
    if not id_token:
        return jsonify({"error": "No token provided"}), 400

    claims = verify_id_token(id_token)
    if not claims:
        return jsonify({"error": "Invalid token"}), 401

    uid = claims.get("uid", "")
    email = claims.get("email", "")
    role = claims.get("role", "")

    # If role not in custom claims, check Firestore
    if not role:
        db = get_db()
        if db:
            user_doc = db.collection("users").document(uid).get()
            if user_doc.exists:
                role = user_doc.to_dict().get("role", "patient")
            else:
                role = "patient"

    full_name = claims.get("name", email.split("@")[0])

    # Check Firestore for full_name
    db = get_db()
    if db:
        user_doc = db.collection("users").document(uid).get()
        if user_doc.exists:
            full_name = user_doc.to_dict().get("full_name", full_name)

    create_session({
        "uid": uid,
        "email": email,
        "role": role,
        "full_name": full_name,
    })

    redirect_url = url_for("doctor.dashboard") if role == "doctor" else url_for("patient.dashboard")
    return jsonify({"success": True, "redirect": redirect_url})


@auth_bp.route("/auth/signup", methods=["POST"])
def handle_signup():
    """
    Server-side signup: create Firebase user + Firestore doc.
    Called from frontend JS after Firebase client-side createUser.
    """
    data = request.get_json(silent=True) or {}
    id_token = data.get("idToken", "")
    role = data.get("role", "patient")
    full_name = data.get("fullName", "")

    if not id_token:
        return jsonify({"error": "No token provided"}), 400

    claims = verify_id_token(id_token)
    if not claims:
        return jsonify({"error": "Invalid token"}), 401

    uid = claims.get("uid", "")
    email = claims.get("email", "")

    # Set custom claims
    try:
        fb_auth.set_custom_user_claims(uid, {"role": role})
    except Exception:
        logger.exception("Failed to set custom claims for uid=%s", uid)

    # Store in Firestore
    db = get_db()
    if db:
        db.collection("users").document(uid).set({
            "uid": uid,
            "role": role,
            "email": email,
            "full_name": full_name or email.split("@")[0],
            "created_at": datetime.utcnow().isoformat(),
        })

    create_session({
        "uid": uid,
        "email": email,
        "role": role,
        "full_name": full_name or email.split("@")[0],
    })

    redirect_url = url_for("doctor.dashboard") if role == "doctor" else url_for("patient.dashboard")
    return jsonify({"success": True, "redirect": redirect_url})


@auth_bp.route("/logout")
def logout():
    clear_session()
    flash("You have been logged out.", "info")
    return redirect(url_for("landing"))
