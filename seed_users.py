"""
CardioSense — Create seed user accounts in Firebase Auth + Firestore.
Run:  python seed_users.py

Creates:
  1. A patient account  (patient@cardiosense.com / Patient@123)
  2. A doctor  account  (doctor@cardiosense.com  / Doctor@123)

Both users are registered in Firebase Auth with custom claims,
and their base records are written to the Firestore 'users' collection.
"""

import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

import config
from services.firebase_service import init_firebase, get_db

import firebase_admin
from firebase_admin import auth as fb_auth


# ── Users to create ──────────────────────────────────────────────────
SEED_USERS = [
    {
        "email": "patient@cardiosense.com",
        "password": "Patient@123",
        "display_name": "John Doe",
        "role": "patient",
    },
    {
        "email": "doctor@cardiosense.com",
        "password": "Doctor@123",
        "display_name": "Dr. Sarah Silva",
        "role": "doctor",
    },
]


def create_user(user_data):
    email = user_data["email"]
    role = user_data["role"]

    # ── 1. Create in Firebase Auth ────────────────────────────────
    try:
        user_record = fb_auth.create_user(
            email=email,
            password=user_data["password"],
            display_name=user_data["display_name"],
        )
        uid = user_record.uid
        print(f"  ✔ Auth account created  →  {email}  (uid: {uid})")
    except firebase_admin.exceptions.AlreadyExistsError:
        # User already exists — look them up
        user_record = fb_auth.get_user_by_email(email)
        uid = user_record.uid
        print(f"  ⚠ Auth account already exists  →  {email}  (uid: {uid})")

    # ── 2. Set custom claims (role) ───────────────────────────────
    fb_auth.set_custom_user_claims(uid, {"role": role})
    print(f"  ✔ Custom claims set  →  role={role}")

    # ── 3. Write to Firestore 'users' collection ──────────────────
    db = get_db()
    if db:
        from datetime import datetime
        user_doc = {
            "uid": uid,
            "email": email,
            "full_name": user_data["display_name"],
            "role": role,
            "created_at": datetime.utcnow().isoformat(),
        }
        db.collection("users").document(uid).set(user_doc, merge=True)
        print(f"  ✔ Firestore doc created  →  users/{uid}")

        # If patient, also create a starter profile
        if role == "patient":
            profile = {
                "uid": uid,
                "full_name": user_data["display_name"],
                "age_years": 50,
                "sex": "Male",
                "height": 175.0,
                "weight": 85.0,
                "bmi": 27.8,
                "ap_hi": 145,
                "ap_lo": 92,
                "cholesterol": 2,
                "gluc": 2,
                "smoke": 0,
                "alco": 0,
                "active": 0,
                "medical_history_text": "Type 2 diabetes diagnosed 2019, mild hypertension",
                "medications_text": "Metformin 500mg twice daily, Losartan 50mg once daily",
                "allergies_text": "Penicillin",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
            db.collection("patient_profiles").document(uid).set(profile, merge=True)
            print(f"  ✔ Patient profile created  →  patient_profiles/{uid}")
    else:
        print("  ✖ Firestore not available — skipped doc creation")

    return uid


def main():
    print("=" * 60)
    print("  CardioSense — Seed Users")
    print("=" * 60)

    # Check service account key
    sa_path = config.FIREBASE_SERVICE_ACCOUNT_JSON_PATH
    if not os.path.isabs(sa_path):
        sa_path = os.path.join(config.BASE_DIR, sa_path)
    if not os.path.exists(sa_path):
        print(f"\n✖ Service account key not found at: {sa_path}")
        print("  Download it from Firebase Console → Project Settings → Service accounts")
        print(f"  Save it as: {config.FIREBASE_SERVICE_ACCOUNT_JSON_PATH}")
        sys.exit(1)

    # Initialize Firebase
    init_firebase()
    print("\n✔ Firebase initialized\n")

    # Create each user
    for user_data in SEED_USERS:
        print(f"\n── Creating {user_data['role']}: {user_data['email']} ──")
        create_user(user_data)

    print("\n" + "=" * 60)
    print("  ✔ Done! You can now log in with:")
    print()
    print("  Patient:  patient@cardiosense.com  /  Patient@123")
    print("  Doctor:   doctor@cardiosense.com   /  Doctor@123")
    print("=" * 60)


if __name__ == "__main__":
    main()
