"""
SLM Summary Service — generates descriptive summaries for Components 1, 2, 3.
Primary: Groq API (if CARDIOSENSE_KEY set).
Fallback: template-based summary.
"""

import json
import logging

from groq import Groq

import config

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None and config.CARDIOSENSE_KEY:
        _client = Groq(api_key=config.CARDIOSENSE_KEY)
    return _client


def _build_prompt(patient_profile, record_values, component_output,
                  component_id):
    """Build the LLM prompt for summarisation."""
    profile_text = (
        f"Patient: {patient_profile.get('full_name', 'Unknown')}, "
        f"Age: {patient_profile.get('age_years', 'N/A')}, "
        f"Sex: {patient_profile.get('sex', 'N/A')}. "
        f"Medical history: {patient_profile.get('medical_history_text', 'None reported')}. "
        f"Medications: {patient_profile.get('medications_text', 'None reported')}. "
        f"Allergies: {patient_profile.get('allergies_text', 'None reported')}."
    )

    record_text = json.dumps(record_values, indent=2) if record_values else "No records."
    output_text = json.dumps(component_output, indent=2, default=str)

    if component_id == 1:
        task = (
            "You are a medical summary writer. Given the patient profile and "
            "cardiovascular risk prediction results, write a concise descriptive "
            "summary in plain language. Do NOT diagnose. Use cautious language "
            "such as 'the model suggests' or 'indicators may point to'. "
            "Mention the risk band (low/medium/high) and probability. "
            "Keep it to 2-3 short paragraphs."
        )
    elif component_id == 2:
        task = (
            "You are a medical summary writer. Given the patient profile and "
            "chest X-ray analysis results (class probabilities and positive "
            "findings), write a concise descriptive summary in plain language. "
            "Do NOT diagnose. Use cautious language. Note any positive findings "
            "and their probabilities. Keep it to 2-3 short paragraphs."
        )
    elif component_id == 3:
        task = (
            "You are a medical summary writer. Given the patient profile and "
            "cardiovascular risk recommendations, write TWO sections:\n"
            "1. A descriptive summary of the risk level and top recommendations "
            "in plain language (2-3 paragraphs).\n"
            "2. A 7-day diet plan with breakfast, lunch, dinner, and a snack "
            "for each day, plus daily hydration reminders. The diet should be "
            "heart-healthy and aligned with the recommendations. "
            "Format the diet plan clearly with day headers.\n"
            "Do NOT diagnose. Use cautious language."
        )
    else:
        task = "Summarize the following clinical data concisely."

    return (
        f"{task}\n\n"
        f"--- Patient Profile ---\n{profile_text}\n\n"
        f"--- Latest Record Values ---\n{record_text}\n\n"
        f"--- Component {component_id} Output ---\n{output_text}"
    )


def _template_summary(patient_profile, component_output, component_id):
    """Fallback template-based summary when Groq is unavailable."""
    name = patient_profile.get("full_name", "The patient")
    age = patient_profile.get("age_years", "N/A")

    if component_id == 1:
        risk = component_output.get("risk_probability", 0)
        band = component_output.get("risk_band", "unknown")
        return (
            f"Based on the submitted clinical data, {name} (age {age}) has an "
            f"estimated cardiovascular risk probability of {risk:.1%}, "
            f"categorised in the {band} risk band. "
            f"This is a model-based estimate and should be interpreted by a "
            f"healthcare professional. Regular monitoring is recommended."
        )

    elif component_id == 2:
        positives = component_output.get("predicted_labels", [])
        top = component_output.get("top_finding", "N/A")
        top_p = component_output.get("top_probability", 0)
        if positives:
            findings = ", ".join(positives)
            return (
                f"The chest X-ray analysis for {name} (age {age}) detected "
                f"potential positive findings: {findings}. The highest "
                f"probability finding is {top} at {top_p:.1%}. "
                f"These results should be reviewed by a radiologist."
            )
        return (
            f"The chest X-ray analysis for {name} (age {age}) did not detect "
            f"any findings above the threshold. The highest probability "
            f"finding is {top} at {top_p:.1%}. "
            f"Results should still be reviewed by a clinician."
        )

    elif component_id == 3:
        risk = component_output.get("risk", 0)
        recs = component_output.get("recommendations", [])
        rec_names = ", ".join(r.get("title", r.get("label", "")) for r in recs[:5])
        summary = (
            f"The cardiovascular risk assessment for {name} (age {age}) "
            f"estimates a risk level of {risk:.1%}. "
            f"Recommended actions include: {rec_names}. "
            f"Please discuss these recommendations with your healthcare provider."
        )
        diet_plan = (
            "\n\n7-Day Heart-Healthy Diet Plan:\n"
            "Day 1: Oatmeal with berries | Grilled chicken salad | "
            "Baked salmon with vegetables | Almonds\n"
            "Day 2: Greek yogurt with nuts | Lentil soup with bread | "
            "Stir-fried tofu with rice | Apple slices\n"
            "Day 3: Whole grain toast with avocado | Quinoa bowl | "
            "Grilled fish with steamed greens | Carrot sticks\n"
            "Day 4: Smoothie with spinach and banana | Turkey wrap | "
            "Chicken breast with sweet potato | Mixed nuts\n"
            "Day 5: Porridge with seeds | Mediterranean salad | "
            "Lean beef with brown rice | Fruit cup\n"
            "Day 6: Eggs with whole grain toast | Chickpea curry | "
            "Baked cod with roasted vegetables | Yogurt\n"
            "Day 7: Banana pancakes | Vegetable soup | "
            "Grilled chicken with quinoa | Dark chocolate (small)\n\n"
            "Hydration: Aim for 8 glasses of water daily."
        )
        return summary + diet_plan

    return "Summary generation is not available for this component."


def generate_summary(patient_profile, record_values, component_output,
                     component_id):
    """
    Generate a descriptive summary for a component result.
    Returns: {"descriptive_summary_text": str, "diet_plan": str (if comp 3)}
    """
    client = _get_client()

    if client:
        prompt = _build_prompt(
            patient_profile, record_values, component_output, component_id
        )
        try:
            resp = client.chat.completions.create(
                model=config.GROQ_CHAT_MODEL,
                messages=[
                    {"role": "system",
                     "content": "You are a cautious medical summary writer. "
                                "Never claim diagnosis. Use language like "
                                "'indicators suggest' or 'the model estimates'."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_completion_tokens=800,
                stream=False,
            )
            text = resp.choices[0].message.content.strip() if resp.choices else ""
            if text:
                result = {"descriptive_summary_text": text}
                if component_id == 3:
                    # For comp 3, the prompt already includes diet plan
                    result["diet_plan"] = text
                return result
        except Exception:
            logger.warning("Groq API call failed for component %s; using template fallback",
                           component_id, exc_info=True)

    # Fallback to template
    text = _template_summary(patient_profile, component_output, component_id)
    result = {"descriptive_summary_text": text}
    if component_id == 3:
        result["diet_plan"] = text
    return result
