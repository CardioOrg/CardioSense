import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Keep feature order identical to training
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

# Default values taken from dataset percentiles to make the form usable without tweaks
DEFAULTS = {
    "age": 53,
    "sex": 0,
    "cp": 1,
    "trestbps": 130,
    "chol": 246,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 1,
    "ca": 1,
    "thal": 2,
}


MODEL_PATH = Path("cardiosense_component1_model.pkl")
DATA_PATH = Path("structured_dataset.csv")


def retrain_and_save():
    """Rebuild the pipeline with the local dataset using the current sklearn version."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)
    X_structured = df[FEATURE_COLUMNS]
    y = df["target"]
    X_text = df["symptom_text"]

    scaler = StandardScaler()
    X_structured_scaled = scaler.fit_transform(X_structured)

    tfidf = TfidfVectorizer(max_features=100)
    X_text_vec = tfidf.fit_transform(X_text)

    combined = hstack([X_structured_scaled, X_text_vec])
    model = LogisticRegression(max_iter=1000)
    model.fit(combined, y)

    joblib.dump(
        {"scaler": scaler, "tfidf_vectorizer": tfidf, "model": model},
        MODEL_PATH,
    )
    return scaler, tfidf, model


@st.cache_resource
def load_artifacts():
    """
    Load scaler, vectorizer and model once for the session.
    If the pickle was created with an incompatible sklearn version, retrain locally.
    """
    if MODEL_PATH.exists():
        package = joblib.load(MODEL_PATH)
        scaler = package["scaler"]
        tfidf = package["tfidf_vectorizer"]
        model = package["model"]

        # Ensure TF-IDF is fitted; mismatched sklearn versions can drop attributes.
        if hasattr(tfidf, "idf_") and getattr(tfidf, "idf_", None) is not None:
            return scaler, tfidf, model, False

    scaler, tfidf, model = retrain_and_save()
    return scaler, tfidf, model, True


def predict_probability(structured_inputs, symptom_text):
    scaler, tfidf, model, retrained = load_artifacts()

    # Use DataFrame to preserve feature names and avoid warnings from scaler
    structured_df = pd.DataFrame([structured_inputs], columns=FEATURE_COLUMNS)
    structured_vector = scaler.transform(structured_df)
    symptom_vector = tfidf.transform([symptom_text])
    combined = hstack([structured_vector, symptom_vector])
    proba = model.predict_proba(combined)[0, 1]
    return float(proba), retrained


st.set_page_config(page_title="CardioSense Risk Estimator", layout="wide")
st.title("CardioSense Component 1 – Symptom Risk Prediction")
st.markdown(
    "Estimate cardiovascular risk by combining clinical factors and a free-text symptom description. "
    "The model mirrors the notebook pipeline (scaled structured features + TF-IDF symptoms -> logistic regression)."
)

with st.form("risk_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=DEFAULTS["age"])
        sex = st.selectbox("Sex (0=female, 1=male)", options=[0, 1], index=DEFAULTS["sex"])
        cp = st.number_input("Chest pain type (0-3)", min_value=0, max_value=3, value=DEFAULTS["cp"])
        trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=80, max_value=200, value=DEFAULTS["trestbps"])
        chol = st.number_input("Serum cholesterol (mg/dl)", min_value=100, max_value=600, value=DEFAULTS["chol"])
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (0=no, 1=yes)", options=[0, 1], index=DEFAULTS["fbs"])
        restecg = st.number_input("Resting ECG results (0-2)", min_value=0, max_value=2, value=DEFAULTS["restecg"])

    with col2:
        thalach = st.number_input("Max heart rate achieved", min_value=60, max_value=220, value=DEFAULTS["thalach"])
        exang = st.selectbox("Exercise induced angina (0=no, 1=yes)", options=[0, 1], index=DEFAULTS["exang"])
        oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=6.0, value=DEFAULTS["oldpeak"], step=0.1)
        slope = st.number_input("Slope of peak exercise ST segment (0-2)", min_value=0, max_value=2, value=DEFAULTS["slope"])
        ca = st.number_input("Number of major vessels colored (0-3)", min_value=0, max_value=3, value=DEFAULTS["ca"])
        thal = st.number_input("Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)", min_value=1, max_value=3, value=DEFAULTS["thal"])
        symptom_text = st.text_area(
            "Symptom description",
            value="chest pain and exercise induced chest pain",
            help="Free-text description is vectorized with TF-IDF (100 features).",
        )

    submitted = st.form_submit_button("Predict risk")

if submitted:
    structured_inputs = [
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal,
    ]
    risk_proba, retrained = predict_probability(structured_inputs, symptom_text)
    risk_percent = round(risk_proba * 100, 1)
    st.success(f"Estimated CVD risk: {risk_percent}%")

    if retrained:
        st.info(" ")

    if risk_proba >= 0.7:
        st.warning("High risk according to the model. Consider medical follow-up.")
    elif risk_proba >= 0.4:
        st.info("Moderate risk. Review clinical context and monitor symptoms.")
    else:
        st.write("Low risk prediction from the model.")

st.caption(
    "Prototype app for demo purposes only. Not a medical device. Uses the trained scaler, TF-IDF vectorizer, and logistic regression model saved in cardiosense_component1_model.pkl."
)
