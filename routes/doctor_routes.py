"""
Doctor routes — dashboard, patient listing, patient detail, bookings, prescriptions.
"""

import json
import logging
from flask import (
    Blueprint, render_template, request, redirect, url_for, flash,
    session, Response
)
from services.auth_service import login_required, role_required
from services.doctor_service import (
    get_all_patients, get_all_doctors, get_patient_detail, get_all_bookings,
    update_booking_status, create_prescription, get_prescription
)
from services.patient_service import get_profile, get_latest_component_run
from services.firebase_service import get_db
from services import pdf_service
from routes.pdf_routes_helper import (
    pdf_response, build_component_pdf, build_combined_pdf,
    build_prescription_pdf, parse_output_json
)

logger = logging.getLogger(__name__)

doctor_bp = Blueprint("doctor", __name__)


@doctor_bp.route("/dashboard")
@login_required
@role_required("doctor")
def dashboard():
    patients = get_all_patients()
    bookings = get_all_bookings()
    pending = [b for b in bookings if b.get("status") == "requested"]
    return render_template("doctor_dashboard.html",
                           patients=patients,
                           bookings=bookings,
                           pending_count=len(pending))


@doctor_bp.route("/patients")
@login_required
@role_required("doctor")
def patient_list():
    patients = get_all_patients()
    return render_template("doctor_patient_list.html", patients=patients)


@doctor_bp.route("/patients/<uid>")
@login_required
@role_required("doctor")
def patient_detail(uid):
    detail = get_patient_detail(uid)
    if not detail or not detail.get("profile"):
        flash("Patient not found.", "warning")
        return redirect(url_for("doctor.patient_list"))

    # Split component runs by component_id for detailed display
    all_runs = detail.get("component_runs", [])

    # Parse output_json if stored as string
    for run in all_runs:
        out = run.get("output_json", {})
        if isinstance(out, str):
            try:
                run["output_json"] = json.loads(out)
            except (json.JSONDecodeError, TypeError):
                run["output_json"] = {}

    comp1_runs = [r for r in all_runs if r.get("component_id") == 1]
    comp2_runs = [r for r in all_runs if r.get("component_id") == 2]
    comp3_runs = [r for r in all_runs if r.get("component_id") == 3]
    comp4_runs = [r for r in all_runs if r.get("component_id") == 4]

    return render_template("doctor_patient_detail.html",
                           detail=detail, patient_uid=uid,
                           comp1_runs=comp1_runs,
                           comp2_runs=comp2_runs,
                           comp3_runs=comp3_runs,
                           comp4_runs=comp4_runs)


@doctor_bp.route("/patients/<uid>/download/component/<int:component_id>")
@login_required
@role_required("doctor")
def download_patient_component_pdf(uid, component_id):
    """Doctor downloads a single component PDF for a patient."""
    pdf_bytes, profile = build_component_pdf(uid, component_id)
    if not pdf_bytes:
        flash("No results found for this component.", "warning")
        return redirect(url_for("doctor.patient_detail", uid=uid))
    name = profile.get("full_name", "patient").replace(" ", "_")
    return pdf_response(
        pdf_bytes, f"{name}_component{component_id}_report.pdf"
    )


@doctor_bp.route("/patients/<uid>/download/combined")
@login_required
@role_required("doctor")
def download_patient_combined_pdf(uid):
    """Doctor downloads combined PDF for a patient."""
    pdf_bytes, profile = build_combined_pdf(uid)
    if not pdf_bytes:
        flash("No results available for report.", "warning")
        return redirect(url_for("doctor.patient_detail", uid=uid))
    name = profile.get("full_name", "patient").replace(" ", "_")
    return pdf_response(pdf_bytes, f"{name}_combined_report.pdf")


@doctor_bp.route("/bookings")
@login_required
@role_required("doctor")
def bookings():
    all_bookings = get_all_bookings()
    return render_template("doctor_bookings.html", bookings=all_bookings)


@doctor_bp.route("/booking/<booking_id>/update", methods=["POST"])
@login_required
@role_required("doctor")
def update_booking(booking_id):
    status = request.form.get("status", "confirmed")
    doctor_uid = session["user"]["uid"]
    update_booking_status(booking_id, status, doctor_uid)
    flash(f"Booking {status}.", "success")
    return redirect(url_for("doctor.bookings"))


@doctor_bp.route("/prescription/create", methods=["GET", "POST"])
@login_required
@role_required("doctor")
def prescription_create():
    doctor_uid = session["user"]["uid"]

    if request.method == "POST":
        data = request.form.to_dict()
        data["doctor_uid"] = doctor_uid

        # Build medication list JSON
        med_names = request.form.getlist("med_name")
        med_dosages = request.form.getlist("med_dosage")
        med_freqs = request.form.getlist("med_frequency")
        meds = []
        for name, dosage, freq in zip(med_names, med_dosages, med_freqs):
            if name.strip():
                meds.append({
                    "medication": name.strip(),
                    "dosage": dosage.strip(),
                    "frequency": freq.strip(),
                })
        data["medication_list_json"] = json.dumps(meds)

        rx = create_prescription(data)
        if rx:
            # Generate and store PDF
            patient_profile = get_profile(data.get("uid", "")) or {}
            doctor_info = {"full_name": session["user"].get("full_name", "")}
            pdf_bytes = pdf_service.generate_prescription_pdf(
                rx, patient_profile, doctor_info
            )
            storage_path = f"prescriptions/{rx['prescription_id']}.pdf"
            pdf_service.save_pdf_to_storage(pdf_bytes, storage_path)

            # Update prescription with PDF path
            db = get_db()
            if db and rx.get("prescription_id"):
                db.collection("prescriptions").document(
                    rx["prescription_id"]
                ).update({"pdf_storage_path": storage_path})

            flash("Prescription created successfully.", "success")
            return redirect(url_for("doctor.prescription_view",
                                    prescription_id=rx["prescription_id"]))

        flash("Error creating prescription.", "danger")
        return redirect(url_for("doctor.prescription_create"))

    # GET: show form with patient/booking selection
    patients = get_all_patients()
    bookings_list = get_all_bookings()
    confirmed = [b for b in bookings_list
                 if b.get("status") in ("confirmed", "requested")]
    return render_template("doctor_prescription_create.html",
                           patients=patients,
                           bookings=confirmed)


@doctor_bp.route("/prescription/<prescription_id>")
@login_required
@role_required("doctor")
def prescription_view(prescription_id):
    rx = get_prescription(prescription_id)
    if not rx:
        flash("Prescription not found.", "warning")
        return redirect(url_for("doctor.dashboard"))

    patient_profile = get_profile(rx.get("uid", "")) or {}

    # Pre-parse medication_list_json for the template
    meds = rx.get("medication_list_json", "[]")
    if isinstance(meds, str):
        try:
            rx["medication_list_json"] = json.loads(meds)
        except (json.JSONDecodeError, TypeError):
            rx["medication_list_json"] = []

    return render_template("doctor_prescription_view.html",
                           prescription=rx,
                           patient=patient_profile)


@doctor_bp.route("/prescription/<prescription_id>/download")
@login_required
@role_required("doctor")
def download_prescription(prescription_id):
    rx = get_prescription(prescription_id)
    if not rx:
        flash("Prescription not found.", "warning")
        return redirect(url_for("doctor.dashboard"))

    patient_profile = get_profile(rx.get("uid", "")) or {}
    doctor_info = {"full_name": session["user"].get("full_name", "")}

    pdf_bytes = pdf_service.generate_prescription_pdf(
        rx, patient_profile, doctor_info
    )
    return pdf_response(pdf_bytes, f"prescription_{prescription_id}.pdf")

