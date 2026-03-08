"""
Component routes — run components 1-4, chatbot interface.
"""

import base64 as b64
import io
import json
import logging

from flask import (
    Blueprint, render_template, request, redirect, url_for, flash,
    session, jsonify
)

import config
from services.auth_service import login_required, role_required
from services.patient_service import (
    get_profile, get_latest_record, save_component_run, add_record
)
from services.component1_service import (
    map_patient_to_features as c1_map, predict_risk
)
from services.component2_service import analyze_cxr
from services.component3_service import (
    map_patient_to_features as c3_map, get_recommendations
)
from services.component4_chat_service import (
    get_or_create_session, send_message, get_session_messages,
    create_new_session
)
from services.slm_summary_service import generate_summary
from services.storage_service import (
    download_blob_bytes, upload_file, upload_bytes_public
)

logger = logging.getLogger(__name__)

component_bp = Blueprint("component", __name__)


@component_bp.route("/1/run", methods=["POST"])
@login_required
@role_required("patient")
def run_component1():
    uid = session["user"]["uid"]
    profile = get_profile(uid) or {}
    record = get_latest_record(uid, "vitals")

    symptom_text = request.form.get(
        "symptom_text", profile.get("medical_history_text", "")
    )

    structured = c1_map(profile, record)
    result = predict_risk(structured, symptom_text)

    # Generate summary
    summary_data = generate_summary(profile, record, result, 1)
    summary_text = summary_data.get("descriptive_summary_text", "")

    save_component_run(
        uid=uid,
        component_id=1,
        input_ref=record.get("id", "") if record else "",
        output_json=result,
        summary_text=summary_text,
    )

    flash("Component 1 analysis complete.", "success")
    return redirect(url_for("patient.component_results"))


@component_bp.route("/2/run", methods=["POST"])
@login_required
@role_required("patient")
def run_component2():
    uid = session["user"]["uid"]
    profile = get_profile(uid) or {}

    # Get the latest CXR record
    cxr_record = get_latest_record(uid, "cxr")
    cxr_path = None
    if cxr_record:
        cxr_path = cxr_record.get("cxr_storage_path", "")

    # Also check if a file was uploaded directly
    image_file = request.files.get("cxr_file")

    if image_file and image_file.filename:

        ext = image_file.filename.rsplit(".", 1)[-1].lower()
        if ext not in config.ALLOWED_IMAGE_EXTENSIONS:
            flash("Invalid image format. Use PNG or JPG.", "danger")
            return redirect(url_for("patient.component_results"))

        path = upload_file(image_file, f"cxr/{uid}")
        add_record(uid, {"record_type": "cxr", "cxr_storage_path": path})

        # Reset file position and analyze
        image_file.seek(0)
        result = analyze_cxr(image_file)
        cxr_path = path

    elif cxr_path:
        # Download from storage and analyze
        try:
            img_bytes = download_blob_bytes(cxr_path)
            if img_bytes:
                result = analyze_cxr(io.BytesIO(img_bytes))
            else:
                flash("Could not retrieve CXR image.", "danger")
                return redirect(url_for("patient.component_results"))
        except Exception:
            logger.exception("CXR analysis failed for uid=%s", uid)
            flash("Error analysing CXR image.", "danger")
            return redirect(url_for("patient.component_results"))
    else:
        flash("No chest X-ray image found. Please upload one first.", "warning")
        return redirect(url_for("patient.records"))

    # Upload images to Firebase Storage and replace base64 with permanent public URLs

    for img_key in ("input_image_b64", "gradcam_image_b64"):
        b64_data = result.get(img_key)
        if b64_data and b64_data.startswith("data:"):
            try:
                # Strip the data URI prefix
                header, encoded = b64_data.split(",", 1)
                raw = b64.b64decode(encoded)
                tag = img_key.replace("_b64", "")
                storage_path = f"component2/{uid}/{tag}.png"
                url = upload_bytes_public(raw, storage_path,
                                          content_type="image/png")
                # Store the permanent public URL
                result[img_key.replace("_b64", "_url")] = url
            except Exception:
                logger.warning("Failed to upload %s for uid=%s", img_key, uid)
        # Remove base64 data so it's not stored in Firestore
        result.pop(img_key, None)

    # Generate summary
    summary_data = generate_summary(profile, {}, result, 2)
    summary_text = summary_data.get("descriptive_summary_text", "")

    save_component_run(
        uid=uid,
        component_id=2,
        input_ref=cxr_path or "",
        output_json=result,
        summary_text=summary_text,
    )

    flash("Component 2 CXR analysis complete.", "success")
    return redirect(url_for("patient.component_results"))


@component_bp.route("/3/run", methods=["POST"])
@login_required
@role_required("patient")
def run_component3():
    uid = session["user"]["uid"]
    profile = get_profile(uid) or {}
    record = get_latest_record(uid, "vitals")

    patient_data = c3_map(profile, record)
    result = get_recommendations(patient_data)

    # Generate summary with diet plan
    summary_data = generate_summary(profile, record, result, 3)
    summary_text = summary_data.get("descriptive_summary_text", "")

    save_component_run(
        uid=uid,
        component_id=3,
        input_ref=record.get("id", "") if record else "",
        output_json=result,
        summary_text=summary_text,
    )

    flash("Component 3 recommendations generated.", "success")
    return redirect(url_for("patient.component_results"))


@component_bp.route("/4/chat")
@login_required
@role_required("patient")
def chat_page():
    uid = session["user"]["uid"]
    sess = get_or_create_session(uid)
    messages = []
    if sess:
        messages = get_session_messages(sess["session_id"])
    return render_template("patient_chat.html",
                           session=sess, messages=messages)


@component_bp.route("/4/chat/send", methods=["POST"])
@login_required
@role_required("patient")
def chat_send():
    uid = session["user"]["uid"]
    data = request.get_json(silent=True) or {}
    user_text = data.get("message", "").strip()
    session_id = data.get("session_id", "")

    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    if not session_id:
        sess = get_or_create_session(uid)
        session_id = sess["session_id"] if sess else ""

    reply = send_message(session_id, uid, user_text)

    # Also save as component run
    save_component_run(
        uid=uid,
        component_id=4,
        input_ref=session_id,
        output_json={"user_message": user_text, "assistant_reply": reply},
        summary_text="",
    )

    return jsonify({"reply": reply, "session_id": session_id})


@component_bp.route("/4/chat/new", methods=["POST"])
@login_required
@role_required("patient")
def chat_new_session():
    uid = session["user"]["uid"]
    sess = create_new_session(uid)
    return redirect(url_for("component.chat_page"))
