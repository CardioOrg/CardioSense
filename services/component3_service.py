"""
Component 3 — Cardiovascular Risk + Recommendation Engine.
Loads calibrated risk model, recommender, and bootstrap models from existing joblib files.
"""
 
from __future__ import annotations
 
import json
import logging
import types
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Sequence
 
import joblib
import numpy as np
import pandas as pd
 
try:
    from sklearn.utils._tags import get_tags
except ImportError:
    get_tags = None
 
import config
 
logger = logging.getLogger(__name__)
 
warnings.filterwarnings(
    "ignore", message=".*serialized model.*", category=UserWarning, module="xgboost"
)
 
# Hypothetical intervention effects for counterfactual risk estimates
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
 
# ── Singleton cache ────────────────────────────────────────────────
_cache = {}
 
 
def _clip_ordinal_1_3(x):
    return int(np.clip(int(round(float(x))), 1, 3))
 
 
def _apply_intervention(x_row: pd.Series, label: str) -> pd.Series:
    x2 = x_row.copy()
    eff = INTERVENTION_EFFECTS.get(label, {})
    for feat, delta in eff.items():
        if feat == "weight_pct":
            continue
        if feat in ("cholesterol", "gluc"):
            x2[feat] = _clip_ordinal_1_3(float(x2.get(feat, 1)) + float(delta))
        elif feat in ("ap_hi", "ap_lo"):
            x2[feat] = float(x2.get(feat, 0)) + float(delta)
        elif feat in ("smoke", "alco", "active"):
            x2[feat] = int(delta)
        else:
            if feat not in x2:
                x2[feat] = delta
    if "weight_pct" in eff and "weight" in x2:
        x2["weight"] = float(x2["weight"]) * (1.0 + float(eff["weight_pct"]))
    if "weight" in x2 and "height" in x2 and float(x2["height"]) > 0:
        h_m = float(x2["height"]) / 100.0
        x2["bmi"] = float(x2["weight"]) / (h_m ** 2)
    if "ap_hi" in x2 and "ap_lo" in x2:
        x2["pulse_pressure"] = float(x2["ap_hi"]) - float(x2["ap_lo"])
    return x2
 
 
def _risk_score(x_row, risk_model, features):
    X1 = pd.DataFrame([{c: x_row.get(c, 0) for c in features}], columns=features)
    return float(risk_model.predict_proba(X1)[:, 1][0])
 
 
def _estimate_risk_drop(x_row, label, risk_model, features):
    base = _risk_score(x_row, risk_model, features)
    new = _risk_score(_apply_intervention(x_row, label), risk_model, features)
    return float(max(0.0, base - new))
 
 
def _add_derived_fields(u: Dict[str, Any]) -> Dict[str, Any]:
    u = dict(u)
    if "bmi" not in u and "weight" in u and "height" in u and float(u["height"]) > 0:
        h_m = float(u["height"]) / 100.0
        u["bmi"] = float(u["weight"]) / (h_m ** 2)
    if "pulse_pressure" not in u and "ap_hi" in u and "ap_lo" in u:
        u["pulse_pressure"] = float(u["ap_hi"]) - float(u["ap_lo"])
    return u
 
 
def _patch_classifier_tags(risk_model):
    try:
        est = risk_model.estimator.estimator
    except Exception:
        return risk_model
    if get_tags:
        try:
            tags = get_tags(est)
            if tags.estimator_type != "classifier":
                patched = replace(tags, estimator_type="classifier")
                est.__sklearn_tags__ = types.MethodType(lambda self: patched, est)
        except Exception:
            est._estimator_type = "classifier"
    else:
        est._estimator_type = "classifier"
    return risk_model
 
 
def _patch_logreg_multi_class(model):
    """Recursively inject multi_class attribute into older LogisticRegression components."""
    if hasattr(model, "steps"):
        for _, step in model.steps:
            if hasattr(step, "estimators_"):
                for est in step.estimators_:
                    if not hasattr(est, "multi_class") and est.__class__.__name__ == "LogisticRegression":
                        est.multi_class = "auto"
            elif not hasattr(step, "multi_class") and step.__class__.__name__ == "LogisticRegression":
                step.multi_class = "auto"
    elif hasattr(model, "estimators_"):
        for est in model.estimators_:
            if not hasattr(est, "multi_class") and est.__class__.__name__ == "LogisticRegression":
                est.multi_class = "auto"
    return model
 
 
def _load_artifacts():
    if "meta" in _cache:
        return _cache
 
    models_dir = config.COMPONENT3_MODELS_DIR
    meta = json.loads(config.COMPONENT3_META_PATH.read_text())
    risk_model = _patch_classifier_tags(
        joblib.load(models_dir / "risk_model_calibrated.joblib")
    )
    rec_model = _patch_logreg_multi_class(joblib.load(models_dir / "recommender_base.joblib"))
    boot_raw = joblib.load(models_dir / "recommender_bootstrap.joblib")
    boot_models = [_patch_logreg_multi_class(m) for m in boot_raw] if boot_raw else []
 
    lib_path = models_dir / "rec_library.json"
    rec_library = json.loads(lib_path.read_text()) if lib_path.exists() else {}
 
    _cache.update(
        meta=meta,
        risk_model=risk_model,
        rec_model=rec_model,
        boot_models=boot_models,
        rec_library=rec_library,
    )
    return _cache
 
 
def map_patient_to_features(profile, record=None):
    """Map patient profile/record to Component 3 feature dict."""
    src = {**(record or {}), **(profile or {})}
    sex_raw = src.get("sex", "")
    gender = 2  # default male
    if str(sex_raw).lower() in ("female", "0", "1"):
        gender = 1
    elif str(sex_raw).lower() in ("male", "1", "2"):
        gender = 2
 
    raw = {
        "gender": gender,
        "height": float(src.get("height", 170)),
        "weight": float(src.get("weight", 70)),
        "ap_hi": float(src.get("ap_hi", 120)),
        "ap_lo": float(src.get("ap_lo", 80)),
        "cholesterol": int(src.get("cholesterol", 1)),
        "gluc": int(src.get("gluc", 1)),
        "smoke": int(src.get("smoke", 0)),
        "alco": int(src.get("alco", 0)),
        "active": int(src.get("active", 0)),
        "age_years": float(src.get("age_years", 50)),
    }
    return raw
 
 
def get_recommendations(patient_data, threshold=0.50, top_k=6):
    """
    Run Component 3.
    patient_data: dict with clinical fields.
    Returns: {"risk": float, "recommendations": [...], "feature_row": {...}}
    """
    artifacts = _load_artifacts()
    features = artifacts["meta"]["features"]
    labels = artifacts["meta"]["recommendation_labels"]
    rec_library = artifacts.get("rec_library", {})
 
    cleaned = _add_derived_fields(patient_data)
    row = {f: cleaned.get(f, 0.0) for f in features}
    x_series = pd.Series(row)
    X_df = pd.DataFrame([row], columns=features)
 
    risk = float(artifacts["risk_model"].predict_proba(X_df)[:, 1][0])
    rec_proba = artifacts["rec_model"].predict_proba(X_df)[0]
 
    boot_probs = None
    if artifacts.get("boot_models"):
        boot_probs = np.vstack(
            [m.predict_proba(X_df)[0] for m in artifacts["boot_models"]]
        )
 
    recs = []
    for idx, label in enumerate(labels):
        p = float(rec_proba[idx])
        if boot_probs is not None:
            low, high = np.percentile(boot_probs[:, idx], [10, 90])
        else:
            low, high = p, p
 
        tpl = rec_library.get(label, {"title": label, "priority": "medium"})
        recs.append({
            "label": label,
            "probability": round(p, 4),
            "confidence_band": (round(float(low), 4), round(float(high), 4)),
            "risk_reduction": round(
                _estimate_risk_drop(x_series, label, artifacts["risk_model"], features), 4
            ),
            "priority": tpl.get("priority", "medium"),
            "title": tpl.get("title", label),
            "why_it_matters": tpl.get("why_it_matters", []),
            "what_to_do": tpl.get("what_to_do", []),
            "targets": tpl.get("targets", []),
            "follow_up": tpl.get("follow_up", ""),
        })
 
    recs = [r for r in recs if r["probability"] >= threshold]
    recs.sort(key=lambda r: (r["probability"], r["risk_reduction"]), reverse=True)
 
    return {
        "risk": round(risk, 4),
        "risk_percent": round(risk * 100, 1),
        "recommendations": recs[:top_k],
        "feature_row": row,
    }
 
 