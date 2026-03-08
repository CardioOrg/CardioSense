"""
Patient routes — dashboard, profile, records, bookings, prescriptions.
"""

import io
import json
import logging

from flask import (
    Blueprint, render_template, request, redirect, url_for, flash,
    session, send_file, Response
)

import config
from services.auth_service import login_required, role_required
from services.patient_service import (
    get_profile, save_profile, add_record, get_records, get_latest_record,
    get_component_runs, get_latest_component_run, create_booking, get_bookings,
    get_prescriptions
)
from services.doctor_service import get_all_doctors
from services.storage_service import upload_file, get_download_url
from services import pdf_service
from routes.pdf_routes_helper import (
    pdf_response, build_component_pdf, build_combined_pdf,
    build_prescription_pdf
)

logger = logging.getLogger(__name__)

patient_bp = Blueprint("patient", __name__)


@patient_bp.route("/dashboard")
@login_required
@role_required("patient")
def dashboard():
    uid = session["user"]["uid"]
    profile = get_profile(uid) or {}
    latest_vitals = get_latest_record(uid, "vitals")
    comp1 = get_latest_component_run(uid, 1)
    comp2 = get_latest_component_run(uid, 2)
    comp3 = get_latest_component_run(uid, 3)
    bookings = get_bookings(uid)
    prescriptions = get_prescriptions(uid)

    return render_template("patient_dashboard.html",
                           profile=profile,
                           latest_vitals=latest_vitals,
                           comp1=comp1, comp2=comp2, comp3=comp3,
                           bookings=bookings,
                           prescriptions=prescriptions)


@patient_bp.route("/profile", methods=["GET", "POST"])
@login_required
@role_required("patient")
def profile():
    uid = session["user"]["uid"]
    if request.method == "POST":
        data = request.form.to_dict()
        save_profile(uid, data)
        flash("Profile saved successfully.", "success")
        return redirect(url_for("patient.profile"))

    profile_data = get_profile(uid) or {}
    return render_template("patient_profile.html", profile=profile_data)


@patient_bp.route("/records", methods=["GET", "POST"])
@login_required
@role_required("patient")
def records():
    uid = session["user"]["uid"]
    if request.method == "POST":
        data = request.form.to_dict()
        record_type = data.get("record_type", "vitals")

        # Handle CXR upload
        if record_type == "cxr" and "cxr_file" in request.files:
            f = request.files["cxr_file"]
            if f and f.filename:
                ext = f.filename.rsplit(".", 1)[-1].lower()
                if ext in config.ALLOWED_IMAGE_EXTENSIONS:
                    path = upload_file(f, f"cxr/{uid}")
                    data["cxr_storage_path"] = path
                else:
                    flash("Invalid image format. Use PNG or JPG.", "danger")
                    return redirect(url_for("patient.records"))

        # Auto-compute BMI
        try:
            h = float(data.get("height", 0))
            w = float(data.get("weight", 0))
            if h > 0 and w > 0:
                data["bmi"] = round(w / ((h / 100.0) ** 2), 1)
        except (ValueError, TypeError):
            pass

        add_record(uid, data)
        flash("Record added successfully.", "success")
        return redirect(url_for("patient.records"))

    all_records = get_records(uid)
    profile = get_profile(uid) or {}
    return render_template("patient_records.html",
                           records=all_records, profile=profile)


@patient_bp.route("/results")
@login_required
@role_required("patient")
def component_results():
    uid = session["user"]["uid"]
    comp1_runs = get_component_runs(uid, 1)
    comp2_runs = get_component_runs(uid, 2)
    comp3_runs = get_component_runs(uid, 3)
    comp4_runs = get_component_runs(uid, 4)
    profile = get_profile(uid) or {}

    return render_template("patient_component_results.html",
                           comp1_runs=comp1_runs,
                           comp2_runs=comp2_runs,
                           comp3_runs=comp3_runs,
                           comp4_runs=comp4_runs,
                           profile=profile)


@patient_bp.route("/booking", methods=["GET", "POST"])
@login_required
@role_required("patient")
def booking():
    uid = session["user"]["uid"]
    if request.method == "POST":
        data = request.form.to_dict()
        create_booking(uid, data)
        flash("Booking request submitted.", "success")
        return redirect(url_for("patient.booking"))

    bookings = get_bookings(uid)
    prescriptions = get_prescriptions(uid)
    doctors = get_all_doctors()
    return render_template("patient_booking.html",
                           bookings=bookings,
                           prescriptions=prescriptions,
                           doctors=doctors)


@patient_bp.route("/download/component/<int:component_id>")
@login_required
@role_required("patient")
def download_component_pdf(component_id):
    uid = session["user"]["uid"]
    pdf_bytes, profile = build_component_pdf(uid, component_id)
    if not pdf_bytes:
        flash("No results found for this component.", "warning")
        return redirect(url_for("patient.component_results"))
    return pdf_response(
        pdf_bytes,
        f"cardiosense_component{component_id}_report.pdf"
    )


@patient_bp.route("/download/combined")
@login_required
@role_required("patient")
def download_combined_pdf():
    uid = session["user"]["uid"]
    pdf_bytes, profile = build_combined_pdf(uid)
    if not pdf_bytes:
        flash("No results available for report.", "warning")
        return redirect(url_for("patient.component_results"))
    return pdf_response(pdf_bytes, "cardiosense_combined_report.pdf")


@patient_bp.route("/download/prescription/<prescription_id>")
@login_required
@role_required("patient")
def download_prescription_pdf(prescription_id):
    uid = session["user"]["uid"]
    pdf_bytes, rx = build_prescription_pdf(prescription_id, requesting_uid=uid)
    if not pdf_bytes:
        flash("Prescription not found.", "warning")
        return redirect(url_for("patient.booking"))
    return pdf_response(pdf_bytes, f"prescription_{prescription_id}.pdf")


@patient_bp.route("/view/prescription/<prescription_id>")
@login_required
@role_required("patient")
def view_prescription_pdf(prescription_id):
    """Open the prescription PDF inline in the browser for printing."""
    uid = session["user"]["uid"]
    pdf_bytes, rx = build_prescription_pdf(prescription_id, requesting_uid=uid)
    if not pdf_bytes:
        flash("Prescription not found.", "warning")
        return redirect(url_for("patient.booking"))
    return pdf_response(
        pdf_bytes, f"prescription_{prescription_id}.pdf", inline=True
    )

