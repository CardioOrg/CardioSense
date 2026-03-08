"""
Component 1 — Symptom & Risk Prediction.
Loads the trained LogisticRegression + TF-IDF pipeline from the existing pkl file.
"""
 
import logging
 
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import config
 
logger = logging.getLogger(__name__)
 
# Feature order must match training
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
 
# ── Singleton model cache ──────────────────────────────────────────
_model_cache = {}
 
 
def _retrain_and_save():
    """Rebuild the pipeline with local dataset (fallback for version mismatch)."""
    data_path = config.COMPONENT1_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")
 
    df = pd.read_csv(data_path)
    X_struct = df[FEATURE_COLUMNS]
    y = df["target"]
    X_text = df["symptom_text"]
 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_struct)
 
    tfidf = TfidfVectorizer(max_features=100)
    X_text_vec = tfidf.fit_transform(X_text)
 
    combined = hstack([X_scaled, X_text_vec])
    model = LogisticRegression(max_iter=1000)
    model.fit(combined, y)
 
    joblib.dump(
        {"scaler": scaler, "tfidf_vectorizer": tfidf, "model": model},
        config.COMPONENT1_MODEL_PATH,
    )
    return scaler, tfidf, model
 
 
def _load_model():
    """Load or retrain the Component 1 model. Cached globally."""
    if "model" in _model_cache:
        return _model_cache["scaler"], _model_cache["tfidf"], _model_cache["model"]
 
    model_path = config.COMPONENT1_MODEL_PATH
    if model_path.exists():
        package = joblib.load(model_path)
        scaler = package["scaler"]
        tfidf = package["tfidf_vectorizer"]
        model = package["model"]
        if hasattr(tfidf, "idf_") and tfidf.idf_ is not None:
            # Fix backward compatibility for older scikit-learn LogisticRegression models
            # In scikit-learn 1.8.0+, 'multi_class' attribute is heavily relied upon, but
            # older pickled models may not have it.
            if not hasattr(model, "multi_class"):
                 model.multi_class = "auto"
           
            _model_cache.update(scaler=scaler, tfidf=tfidf, model=model)
            return scaler, tfidf, model
 
    scaler, tfidf, model = _retrain_and_save()
    if not hasattr(model, "multi_class"):
         model.multi_class = "auto"
    _model_cache.update(scaler=scaler, tfidf=tfidf, model=model)
    return scaler, tfidf, model
 
 
def map_patient_to_features(profile, record=None):
    """
    Map patient profile / record fields to the 13 structured features
    expected by Component 1.
    """
    src = {**(record or {}), **(profile or {})}
    sex_map = {"male": 1, "female": 0, "Male": 1, "Female": 0}
    sex_raw = src.get("sex", 0)
    sex = sex_map.get(str(sex_raw), int(sex_raw) if str(sex_raw).isdigit() else 0)
 
    return [
        int(src.get("age_years", src.get("age", 50))),
        sex,
        int(src.get("cp", 1)),
        int(src.get("ap_hi", src.get("trestbps", 130))),
        int(src.get("cholesterol", src.get("chol", 200))),
        int(src.get("fbs", 0)),
        int(src.get("restecg", 0)),
        int(src.get("thalach", 150)),
        int(src.get("exang", 0)),
        float(src.get("oldpeak", 1.0)),
        int(src.get("slope", 1)),
        int(src.get("ca", 1)),
        int(src.get("thal", 2)),
    ]
 
 
def predict_risk(structured_features, symptom_text=""):
    """
    Run Component 1 prediction.
    Returns: {"risk_probability": float, "risk_band": str}
    """
    scaler, tfidf, model = _load_model()
 
    structured_df = pd.DataFrame([structured_features], columns=FEATURE_COLUMNS)
    scaled = scaler.transform(structured_df)
    text_vec = tfidf.transform([symptom_text or ""])
    combined = hstack([scaled, text_vec])
 
    proba = float(model.predict_proba(combined)[0, 1])
 
    if proba >= 0.7:
        band = "high"
    elif proba >= 0.4:
        band = "medium"
    else:
        band = "low"
 
    return {
        "risk_probability": round(proba, 4),
        "risk_percent": round(proba * 100, 1),
        "risk_band": band,
    }
 
 