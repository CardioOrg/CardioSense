
from __future__ import annotations

import json
import types
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Sequence

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.utils._tags import get_tags


warnings.filterwarnings(
    "ignore",
    message=".*serialized model.*",
    category=UserWarning,
    module="xgboost",
)

# Hypothetical intervention effects used for counterfactual risk estimates.
INTERVENTION_EFFECTS = {
    "diet_low_sodium": {"ap_hi": -5.0, "ap_lo": -3.0},
    "diet_mediterranean": {"cholesterol": -1},
    "diet_reduce_added_sugar": {"gluc": -1},
    "activity_aerobic": {"ap_hi": -3.0, "ap_lo": -2.0, "weight_pct": -0.02, "active": 1},
    "activity_strength": {"weight_pct": -0.01, "active": 1},
    "weight_loss_plan": {"weight_pct": -0.05},
    "sleep_hygiene": {"ap_hi": -2.0, "ap_lo": -1.0},
    "stress_reduction": {"ap_hi": -4.0, "ap_lo": -2.0},
    "smoking_cessation": {"smoke": 0},
    "alcohol_reduction": {"alco": 0},
    "clinician_followup_routine": {},
    "clinician_followup_urgent": {},
    "bp_home_monitoring": {},
    "lipid_followup": {},
    "glucose_followup": {},
}


def _clip_ordinal_1_3(x: Any) -> int:
    return int(np.clip(int(round(float(x))), 1, 3))


def _apply_intervention(x_row: pd.Series, label: str) -> pd.Series:
    """Simulate a modest effect of the recommendation on the feature vector."""
    x2 = x_row.copy()
    eff = INTERVENTION_EFFECTS.get(label, {})

    for feat, delta in eff.items():
        if feat == "weight_pct":
            continue

        if feat in ("cholesterol", "gluc"):
            x2[feat] = _clip_ordinal_1_3(float(x2.get(feat, 1)) + float(delta))
        elif feat in ("ap_hi", "ap_lo"):
            x2[feat] = float(x2.get(feat, 0.0)) + float(delta)
        elif feat in ("smoke", "alco", "active"):
            x2[feat] = int(delta)
        else:
            if feat not in x2:
                x2[feat] = delta

    if "weight_pct" in eff and "weight" in x2:
        x2["weight"] = float(x2["weight"]) * (1.0 + float(eff["weight_pct"]))

    if ("weight" in x2) and ("height" in x2) and float(x2["height"]) > 0:
        h_m = float(x2["height"]) / 100.0
        x2["bmi"] = float(x2["weight"]) / (h_m**2)

    if ("ap_hi" in x2) and ("ap_lo" in x2):
        x2["pulse_pressure"] = float(x2["ap_hi"]) - float(x2["ap_lo"])

    return x2


def _risk_score(x_row: pd.Series, risk_model: Any, features: Sequence[str]) -> float:
    X1 = pd.DataFrame([{c: x_row.get(c, 0) for c in features}], columns=features)
    return float(risk_model.predict_proba(X1)[:, 1][0])


def _estimate_risk_drop(
    x_row: pd.Series,
    label: str,
    risk_model: Any,
    features: Sequence[str],
) -> float:
    base = _risk_score(x_row, risk_model, features)
    new = _risk_score(_apply_intervention(x_row, label), risk_model, features)
    return float(max(0.0, base - new))


def _add_derived_fields(u: Dict[str, Any]) -> Dict[str, Any]:
    u = dict(u)
    if "bmi" not in u and "weight" in u and "height" in u and float(u["height"]) > 0:
        h_m = float(u["height"]) / 100.0
        u["bmi"] = float(u["weight"]) / (h_m**2)
    if "pulse_pressure" not in u and "ap_hi" in u and "ap_lo" in u:
        u["pulse_pressure"] = float(u["ap_hi"]) - float(u["ap_lo"])
    return u


def _patch_classifier_tags(risk_model: Any) -> Any:
    """Add a classifier tag to the frozen XGBoost estimator so predict_proba works."""
    try:
        est = risk_model.estimator.estimator
    except Exception:
        return risk_model

    try:
        tags = get_tags(est)
        if tags.estimator_type != "classifier":
            patched = replace(tags, estimator_type="classifier")
            est.__sklearn_tags__ = types.MethodType(lambda self: patched, est)
    except Exception:
        est._estimator_type = "classifier"  # best-effort fallback
    return risk_model


def _load_rec_library(base_dir: Path) -> Dict[str, Dict[str, Any]]:
    lib_paths = [
        base_dir / "models" / "rec_library.json",
        base_dir / "data" / "rec_library.json",
    ]
    for p in lib_paths:
        if p.exists():
            return json.loads(p.read_text())
    return {}


@st.cache_resource(show_spinner=False)
def load_artifacts():
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models"

    meta = json.loads((base_dir / "meta.json").read_text())
    risk_model = _patch_classifier_tags(joblib.load(models_dir / "risk_model_calibrated.joblib"))
    rec_model = joblib.load(models_dir / "recommender_base.joblib")
    boot_models = joblib.load(models_dir / "recommender_bootstrap.joblib")

    return {
        "meta": meta,
        "risk_model": risk_model,
        "rec_model": rec_model,
        "boot_models": boot_models,
        "rec_library": _load_rec_library(base_dir),
    }


def prepare_feature_row(raw: Dict[str, Any], feature_order: Sequence[str]) -> Dict[str, Any]:
    """Cast inputs to numeric, add derived metrics, and align to the trained feature list."""
    cleaned: Dict[str, Any] = {}
    int_fields = {"gender", "cholesterol", "gluc", "smoke", "alco", "active"}

    for k, v in raw.items():
        if k in int_fields:
            cleaned[k] = int(v)
        else:
            cleaned[k] = float(v)

    cleaned = _add_derived_fields(cleaned)
    return {f: cleaned.get(f, 0.0) for f in feature_order}


def recommend_real_world(
    user_input: Dict[str, Any],
    artifacts: Dict[str, Any],
    threshold: float = 0.50,
    top_k: int = 6,
) -> Dict[str, Any]:
    features = artifacts["meta"]["features"]
    labels = artifacts["meta"]["recommendation_labels"]
    rec_library = artifacts.get("rec_library", {})

    row = prepare_feature_row(user_input, features)
    x_series = pd.Series(row)
    X_df = pd.DataFrame([row], columns=features)

    risk = float(artifacts["risk_model"].predict_proba(X_df)[:, 1][0])
    rec_proba = artifacts["rec_model"].predict_proba(X_df)[0]

    boot_probs = None
    if artifacts.get("boot_models"):
        boot_probs = np.vstack([m.predict_proba(X_df)[0] for m in artifacts["boot_models"]])

    recs = []
    for idx, label in enumerate(labels):
        p = float(rec_proba[idx])
        if boot_probs is not None:
            low, high = np.percentile(boot_probs[:, idx], [10, 90])
        else:
            low, high = p, p

        tpl = rec_library.get(label, {}) if rec_library else {}
        if not tpl:
            tpl = {"title": label, "priority": "medium"}
        recs.append(
            {
                "label": label,
                "probability": p,
                "confidence_band": (float(low), float(high)),
                "risk_reduction": _estimate_risk_drop(x_series, label, artifacts["risk_model"], features),
                "priority": tpl.get("priority", "medium"),
                "title": tpl.get("title", label),
                "why_it_matters": [s.format(**row) for s in tpl.get("why_it_matters", [])],
                "what_to_do": [s.format(**row) for s in tpl.get("what_to_do", [])],
                "targets": [s.format(**row) for s in tpl.get("targets", [])],
                "follow_up": tpl.get("follow_up", ""),
            }
        )

    recs = [r for r in recs if r["probability"] >= threshold]
    recs.sort(key=lambda r: (r["probability"], r["risk_reduction"]), reverse=True)
    return {"risk": risk, "recommendations": recs[:top_k], "feature_row": row}


def main() -> None:
    st.set_page_config(page_title="CardioSense — Component 3", layout="wide")
    st.title("CardioSense — Cardiovascular Risk + Recommendations")
    st.caption("Powered by pretrained models from the Component 3 notebook (Kaggle cardiovascular dataset).")

    artifacts = load_artifacts()

    default_user = {
        "age_years": 55,
        "gender": 2,
        "height": 170,
        "weight": 86,
        "ap_hi": 155,
        "ap_lo": 95,
        "cholesterol": 3,
        "gluc": 2,
        "smoke": 1,
        "alco": 0,
        "active": 0,
    }

    with st.form("user-input"):
        st.subheader("Patient profile")
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox(
                "Sex (1=female, 2=male)",
                options=[1, 2],
                index=1 if default_user["gender"] == 2 else 0,
            )
            age_years = st.slider("Age (years)", 18, 90, int(default_user["age_years"]))
            active = st.checkbox("Physically active weekly", value=bool(default_user["active"]))

        with c2:
            height = st.slider("Height (cm)", 120, 220, int(default_user["height"]))
            weight = st.slider("Weight (kg)", 35, 200, int(default_user["weight"]))
            smoke = st.checkbox("Currently smokes", value=bool(default_user["smoke"]))
            alco = st.checkbox("Regular alcohol use", value=bool(default_user["alco"]))

        with c3:
            ap_hi = st.slider("Systolic BP (ap_hi)", 80, 220, int(default_user["ap_hi"]))
            ap_lo = st.slider("Diastolic BP (ap_lo)", 40, 160, int(default_user["ap_lo"]))
            cholesterol = st.selectbox(
                "Cholesterol category (1=normal, 2=above normal, 3=well above)",
                options=[1, 2, 3],
                index=default_user["cholesterol"] - 1,
            )
            gluc = st.selectbox(
                "Glucose category (1=normal, 2=above normal, 3=well above)",
                options=[1, 2, 3],
                index=default_user["gluc"] - 1,
            )

        cols = st.columns(3)
        with cols[0]:
            threshold = st.slider("Recommendation probability cutoff", 0.10, 0.90, 0.50, 0.05)
        with cols[1]:
            top_k = st.slider("Max recommendations to show", 3, 10, 6, 1)

        submitted = st.form_submit_button("Run models")

    if submitted:
        user_input = {
            "gender": gender,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": int(smoke),
            "alco": int(alco),
            "active": int(active),
            "age_years": age_years,
        }

        output = recommend_real_world(user_input, artifacts, threshold=threshold, top_k=top_k)
        derived = _add_derived_fields(user_input)

        st.subheader("Risk estimate")
        risk_cols = st.columns(2)
        with risk_cols[0]:
            st.metric("Estimated cardiovascular risk", f"{output['risk'] * 100:.1f}%")
        with risk_cols[1]:
            st.write("Derived metrics")
            st.write(
                f"BMI: **{derived.get('bmi', 0):.1f}**  |  "
                f"Pulse pressure: **{derived.get('pulse_pressure', 0):.0f} mmHg**"
            )

        st.subheader("Personalized recommendations")
        if not output["recommendations"]:
            st.info("No recommendations cleared the probability threshold. Lower the cutoff to see more.")
        else:
            for rec in output["recommendations"]:
                prob_pct = f"{rec['probability'] * 100:.0f}%"
                low, high = rec["confidence_band"]
                conf_txt = f"{low * 100:.0f}%–{high * 100:.0f}%"
                delta_txt = f"{rec['risk_reduction'] * 100:.1f} pts"

                with st.expander(f"{rec['title']}  •  {prob_pct} (conf {conf_txt})"):
                    st.write(f"Priority: **{rec['priority'].title()}**")
                    st.write(f"Estimated risk drop if followed: **{delta_txt}**")
                    if rec["why_it_matters"]:
                        st.write("Why it matters:")
                        for line in rec["why_it_matters"]:
                            st.write(f"- {line}")
                    if rec["what_to_do"]:
                        st.write("What to do next:")
                        for line in rec["what_to_do"]:
                            st.write(f"- {line}")
                    if rec["targets"]:
                        st.write("Targets:")
                        for line in rec["targets"]:
                            st.write(f"- {line}")
                    if rec["follow_up"]:
                        st.write(f"Follow-up: {rec['follow_up']}")
    else:
        st.info("Fill in the patient profile and click **Run models** to see risk and recommendations.")


if __name__ == "__main__":
    main()
