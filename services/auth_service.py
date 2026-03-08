"""
Auth service — Firebase ID token verification, session management, role decorators.
"""
 
import functools
import logging
from datetime import datetime
 
from flask import session, redirect, url_for, flash, abort, request
from firebase_admin import auth as fb_auth
 
from services.firebase_service import get_db
 
logger = logging.getLogger(__name__)
 
 
def verify_id_token(id_token):
    """Verify a Firebase ID token. Returns decoded claims dict or None."""
    try:
        decoded = fb_auth.verify_id_token(id_token)
        return decoded
    except Exception as e:
        logger.error("Error verifying ID token: %s", e)
        return None
 
 
def create_session(user_info):
    """Store user info in Flask session."""
    session["user"] = {
        "uid": user_info.get("uid", ""),
        "email": user_info.get("email", ""),
        "role": user_info.get("role", "patient"),
        "full_name": user_info.get("full_name", ""),
    }
 
 
def clear_session():
    """Remove user data from Flask session."""
    session.pop("user", None)
 
 
def get_current_user():
    """Return the current user dict from session, or None."""
    return session.get("user")
 
 
def signup_user(email, password, full_name, role):
    """
    Create a Firebase Auth user, set custom claims, and store in Firestore.
    Returns (user_record, error_message).
    """
    try:
        user = fb_auth.create_user(email=email, password=password,
                                   display_name=full_name)
        # Set custom claims for role-based access
        fb_auth.set_custom_user_claims(user.uid, {"role": role})
 
        # Store user document in Firestore
        db = get_db()
        if db:
            db.collection("users").document(user.uid).set({
                "uid": user.uid,
                "role": role,
                "email": email,
                "full_name": full_name,
                "created_at": datetime.utcnow().isoformat(),
            })
 
        return user, None
    except Exception as e:
        return None, str(e)
 
 
# ── Decorators ─────────────────────────────────────────────────────
 
def login_required(f):
    """Decorator: redirect to login if no session."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated
 
 
def role_required(role):
    """Decorator factory: restrict access to a specific role."""
    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            user = session.get("user")
            if not user:
                flash("Please log in to continue.", "warning")
                return redirect(url_for("auth.login"))
            if user.get("role") != role:
                flash("You do not have permission to access this page.", "danger")
                return redirect(url_for("landing"))
            return f(*args, **kwargs)
        return decorated
    return decorator
 