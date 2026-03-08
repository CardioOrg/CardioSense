"""
PDF Service -- generate per-component and combined PDF reports, plus prescription PDFs.
Uses fpdf2.
"""

import io
import json
import os
import re
import tempfile
from datetime import datetime
from fpdf import FPDF
from services import storage_service


def _sanitize(text):
    """Replace Unicode characters that Helvetica cannot render with ASCII equivalents."""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "\u2014": "-",   # em-dash
        "\u2013": "-",   # en-dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u2022": "*",   # bullet
        "\u00b7": "*",   # middle dot
        "\u2032": "'",   # prime
        "\u2033": '"',   # double prime
        "\u00a0": " ",   # non-breaking space
        "\u2212": "-",   # minus sign
        "\u00d7": "x",   # multiplication sign
        "\u00f7": "/",   # division sign
        "\u2192": "->",  # right arrow
        "\u2190": "<-",  # left arrow
        "\u2264": "<=",  # less than or equal
        "\u2265": ">=",  # greater than or equal
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Remove any remaining non-latin1 characters
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text


class CardioSensePDF(FPDF):
    """Custom PDF with CardioSense branding."""

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(220, 53, 69)
        self.cell(0, 10, "CardioSense", border=0, ln=False, align="L")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                  border=0, ln=True, align="R")
        self.line(10, 22, 200, 22)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(
            0, 10,
            "CardioSense Report - For informational purposes only. "
            "Not a medical diagnosis.",
            border=0, align="C",
        )

    def section_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(33, 37, 41)
        self.cell(0, 8, _sanitize(title), ln=True)
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5, _sanitize(text))
        self.ln(3)

    def key_value(self, key, value):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 60)
        self.cell(55, 6, _sanitize(f"{key}:"), ln=False)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, _sanitize(str(value)), ln=True)


def _patient_header(pdf, patient_info):
    """Add patient info section to PDF."""
    pdf.section_title("Patient Information")
    pdf.key_value("Name", patient_info.get("full_name", "N/A"))
    pdf.key_value("Age", patient_info.get("age_years", "N/A"))
    pdf.key_value("Sex", patient_info.get("sex", "N/A"))
    pdf.key_value("Date", datetime.utcnow().strftime("%Y-%m-%d"))
    pdf.ln(5)


def _render_component2_visuals(pdf, result_data):
    """Embed CXR/Grad-CAM images and a bar chart into the PDF for Component 2."""
    tmp_files = []
    try:
        img_url_input = result_data.get("input_image_url")
        img_url_gradcam = result_data.get("gradcam_image_url")

        if img_url_input or img_url_gradcam:
            pdf.ln(3)
            start_y = pdf.get_y()
            img_w = 85

            if img_url_input:
                try:
                    import urllib.request
                    tmp_in = tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False)
                    tmp_files.append(tmp_in.name)
                    urllib.request.urlretrieve(img_url_input, tmp_in.name)
                    tmp_in.close()
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.cell(img_w, 5, "Input Chest X-Ray", ln=True)
                    pdf.image(tmp_in.name, x=10, w=img_w)
                except Exception:
                    pass

            if img_url_gradcam:
                try:
                    import urllib.request
                    tmp_gc = tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False)
                    tmp_files.append(tmp_gc.name)
                    urllib.request.urlretrieve(img_url_gradcam, tmp_gc.name)
                    tmp_gc.close()
                    pdf.set_y(start_y)
                    pdf.set_x(105)
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.cell(img_w, 5,
                             _sanitize(f"Grad-CAM ({result_data.get('top_finding', '')})"))
                    pdf.image(tmp_gc.name, x=105, y=start_y + 6, w=img_w)
                except Exception:
                    pass

            pdf.set_y(max(pdf.get_y(), start_y + 70))
            pdf.ln(5)

        # -- Generate bar chart with matplotlib --
        all_results = result_data.get("all_results", [])
        if all_results:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                chart_labels = [r[0] for r in reversed(all_results)]
                chart_probs = [r[1] for r in reversed(all_results)]
                threshold = result_data.get("threshold", 0.7)
                colors = []
                for p in chart_probs:
                    if p >= 0.7:
                        colors.append("#dc3545")
                    elif p >= 0.5:
                        colors.append("#ffc107")
                    else:
                        colors.append("#6c757d")

                fig, ax = plt.subplots(figsize=(7, 4.5))
                bars = ax.barh(chart_labels, chart_probs, color=colors,
                               edgecolor="white", height=0.7)
                ax.axvline(x=threshold, color="#dc3545", linestyle="--",
                           linewidth=1.5, label=f"{threshold:.0%} threshold")
                ax.set_xlim(0, 1.0)
                ax.set_xlabel("Probability")
                ax.set_title("CXR Analysis Results")
                ax.legend(loc="lower right")

                for bar_item, prob in zip(bars, chart_probs):
                    ax.text(prob + 0.01,
                            bar_item.get_y() + bar_item.get_height() / 2,
                            f"{prob:.1%}", va="center", fontsize=7)

                plt.tight_layout()

                tmp_chart = tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False)
                tmp_files.append(tmp_chart.name)
                fig.savefig(tmp_chart.name, dpi=150, bbox_inches="tight")
                plt.close(fig)
                tmp_chart.close()

                if pdf.get_y() > 160:
                    pdf.add_page()

                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 6, "Findings Chart", ln=True)
                pdf.ln(2)
                pdf.image(tmp_chart.name, x=10, w=190)
            except Exception:
                # Fallback to table
                pdf.ln(2)
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(90, 6, "Finding", border=1)
                pdf.cell(40, 6, "Probability", border=1, ln=True)
                pdf.set_font("Helvetica", "", 9)
                for label, prob in all_results:
                    pdf.cell(90, 6, _sanitize(label), border=1)
                    pdf.cell(40, 6, f"{prob:.4f}", border=1, ln=True)

    finally:
        for f in tmp_files:
            try:
                os.unlink(f)
            except OSError:
                pass


def generate_component_pdf(patient_info, component_id, result_data,
                           summary_text=""):
    """
    Generate a PDF for a single component result.
    Returns PDF bytes.
    """
    pdf = CardioSensePDF()
    pdf.add_page()
    _patient_header(pdf, patient_info)

    titles = {
        1: "Component 1 - Symptom & Risk Prediction",
        2: "Component 2 - Chest X-Ray Analysis",
        3: "Component 3 - Recommendations & Risk Assessment",
        4: "Component 4 - Chatbot Session",
    }
    pdf.section_title(titles.get(component_id, f"Component {component_id}"))

    if component_id == 1:
        pdf.key_value("Risk Probability",
                      f"{result_data.get('risk_percent', 0)}%")
        pdf.key_value("Risk Band", result_data.get("risk_band", "N/A"))

    elif component_id == 2:
        pdf.key_value("Positive Findings",
                      str(result_data.get("positive_findings", 0)))
        pdf.key_value("Top Finding", result_data.get("top_finding", "N/A"))
        pdf.key_value("Top Probability",
                      f"{result_data.get('top_probability', 0):.1%}")
        pdf.key_value("Threshold", f"{result_data.get('threshold', 0.7):.0%}")
        labels = result_data.get("predicted_labels", [])
        if labels:
            pdf.body_text(f"Detected: {', '.join(labels)}")

        _render_component2_visuals(pdf, result_data)

    elif component_id == 3:
        pdf.key_value("Estimated Risk",
                      f"{result_data.get('risk_percent', 0)}%")
        recs = result_data.get("recommendations", [])
        if recs:
            pdf.ln(3)
            pdf.section_title("Recommendations")
            for i, rec in enumerate(recs, 1):
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 6,
                         _sanitize(f"{i}. {rec.get('title', rec.get('label', ''))} "
                         f"({rec.get('probability', 0):.0%})"),
                         ln=True)
                pdf.set_font("Helvetica", "", 9)
                if rec.get("priority"):
                    pdf.cell(0, 5, f"   Priority: {rec['priority']}", ln=True)
                rr = rec.get("risk_reduction", 0)
                if rr:
                    pdf.cell(0, 5,
                             f"   Est. risk reduction: {rr:.1%} points",
                             ln=True)
                pdf.ln(2)

    elif component_id == 4:
        messages = result_data.get("messages", [])
        for msg in messages:
            role_label = "Patient" if msg.get("role") == "user" else "CardioSense"
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 5, f"{role_label}:", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, _sanitize(msg.get("content", "")))
            pdf.ln(1)

    # Summary
    if summary_text:
        pdf.ln(3)
        pdf.section_title("AI-Generated Summary")
        pdf.body_text(summary_text)

    # Disclaimer
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(
        0, 4,
        "Disclaimer: This report is generated by CardioSense, a research "
        "project prototype. It is not a medical diagnosis and should not "
        "replace professional medical advice."
    )

    return pdf.output()


def generate_combined_pdf(patient_info, all_results):
    """
    Generate a combined PDF with all component results.
    all_results: dict {component_id: result_data}
    Returns PDF bytes.
    """
    pdf = CardioSensePDF()
    pdf.add_page()
    _patient_header(pdf, patient_info)
    pdf.section_title("Combined CardioSense Report")
    pdf.body_text(
        "This report combines the latest results from all CardioSense "
        "components for the above patient."
    )

    for comp_id in [1, 2, 3, 4]:
        data = all_results.get(comp_id)
        if not data:
            continue
        pdf.add_page()
        titles = {
            1: "Component 1 - Symptom & Risk Prediction",
            2: "Component 2 - Chest X-Ray Analysis",
            3: "Component 3 - Recommendations",
            4: "Component 4 - Chat Summary",
        }
        pdf.section_title(titles.get(comp_id, f"Component {comp_id}"))

        output = data.get("output_json", {})
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except (json.JSONDecodeError, TypeError):
                output = {}

        if comp_id == 1:
            pdf.key_value("Risk", f"{output.get('risk_percent', 0)}%")
            pdf.key_value("Band", output.get("risk_band", "N/A"))
        elif comp_id == 2:
            pdf.key_value("Positive Findings",
                          str(output.get("positive_findings", 0)))
            pdf.key_value("Top Finding", output.get("top_finding", "N/A"))
            pdf.key_value("Threshold", f"{output.get('threshold', 0.7):.0%}")
            _render_component2_visuals(pdf, output)
        elif comp_id == 3:
            pdf.key_value("Risk", f"{output.get('risk_percent', 0)}%")
            recs = output.get("recommendations", [])
            for r in recs[:5]:
                pdf.body_text(
                    f"• {r.get('title', r.get('label', ''))} "
                    f"({r.get('probability', 0):.0%})"
                )

        summary = data.get("summary_text", "")
        if summary:
            pdf.ln(2)
            pdf.section_title("Summary")
            pdf.body_text(summary)

    return pdf.output()


def generate_prescription_pdf(prescription, patient_info, doctor_info):
    """Generate a prescription PDF."""
    pdf = CardioSensePDF()
    pdf.add_page()

    pdf.section_title("Medical Prescription")
    pdf.ln(3)

    pdf.key_value("Patient", patient_info.get("full_name", "N/A"))
    pdf.key_value("Age", str(patient_info.get("age_years", "N/A")))
    pdf.key_value("Prescribed by", doctor_info.get("full_name", "N/A"))
    pdf.key_value("Date", prescription.get("created_at", "N/A")[:10])
    pdf.ln(5)

    # Medications
    pdf.section_title("Medications")
    meds = prescription.get("medication_list_json", "[]")
    if isinstance(meds, str):
        try:
            meds = json.loads(meds)
        except (json.JSONDecodeError, TypeError):
            meds = [{"medication": meds}]

    if isinstance(meds, list):
        for i, med in enumerate(meds, 1):
            if isinstance(med, dict):
                name = med.get("medication", med.get("name", ""))
                dosage = med.get("dosage", "")
                freq = med.get("frequency", "")
                pdf.body_text(f"{i}. {name} - {dosage} - {freq}")
            else:
                pdf.body_text(f"{i}. {med}")

    # Instructions
    instructions = prescription.get("instructions_text", "")
    if instructions:
        pdf.ln(3)
        pdf.section_title("Instructions")
        pdf.body_text(instructions)

    # Follow-up
    followup = prescription.get("followup_text", "")
    if followup:
        pdf.ln(3)
        pdf.section_title("Follow-up")
        pdf.body_text(followup)

    return pdf.output()


def save_pdf_to_storage(pdf_bytes, storage_path):
    """Upload PDF bytes to Cloud Storage and return the path."""
    try:
        return storage_service.upload_bytes(pdf_bytes, storage_path)
    except Exception:
        return None
