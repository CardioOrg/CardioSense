# CardioSense

**Cardiovascular Health Management System** — An AI-powered web application for managing cardiovascular patient data, running trained ML models for risk prediction, CXR analysis, and health recommendations, plus a supportive health chatbot.

## Features

| Feature | Description |
|---------|-------------|
| **Risk Prediction** (C1) | Symptom-based cardiovascular risk assessment using clinical features + text |
| **CXR Analysis** (C2) | Chest X-ray triage with DenseNet-121 (14 thoracic disease labels) |
| **Recommendations** (C3) | Personalised lifestyle recommendations with counterfactual risk reduction |
| **Health Chatbot** (C4) | Supportive companion with emergency/self-harm safety routing |
| **Doctor Portal** | Patient listing, booking management, prescription creation |
| **Patient Portal** | Profile, records, component results, PDF reports, doctor channeling |

## Tech Stack

- **Backend:** Python Flask
- **Database:** Firebase Firestore
- **Auth:** Firebase Authentication (email/password) + Custom Claims
- **Storage:** Firebase Cloud Storage
- **Frontend:** Jinja2 templates + Bootstrap 5.3
- **ML:** PyTorch (DenseNet-121), scikit-learn, XGBoost
- **LLM:** Groq API (Llama 4 Scout)
- **PDF:** fpdf2

## Quick Start

```bash
# 1. Clone and enter
cd CardioSenseApp

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env with your Firebase and Groq credentials

# 5. Run
python app.py
```

Open http://localhost:5000

## Environment Variables

| Variable | Description |
|----------|-------------|
| `FLASK_SECRET_KEY` | Flask session secret key |
| `FIREBASE_PROJECT_ID` | Firebase project ID |
| `FIREBASE_SERVICE_ACCOUNT_JSON_PATH` | Path to Firebase service account JSON |
| `FIREBASE_STORAGE_BUCKET` | Firebase Cloud Storage bucket |
| `FIREBASE_API_KEY` | Firebase Web API key (for client-side auth) |
| `CARDIOSENSE_KEY` | Groq API key for chatbot and summaries |

## Project Structure

```
CardioSenseApp/
├── app.py                    # Flask app factory
├── config.py                 # Configuration & env vars
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── services/                 # Business logic layer
│   ├── auth_service.py       # Firebase Auth + RBAC
│   ├── firebase_service.py   # Firebase init
│   ├── storage_service.py    # Cloud Storage helpers
│   ├── patient_service.py    # Patient CRUD
│   ├── doctor_service.py     # Doctor operations
│   ├── component1_service.py # Risk prediction
│   ├── component2_service.py # CXR analysis
│   ├── component3_service.py # Recommendations
│   ├── component4_chat_service.py  # Chatbot
│   ├── slm_summary_service.py     # AI summaries
│   └── pdf_service.py        # PDF report generation
├── routes/                   # Flask blueprints
│   ├── auth_routes.py        # Login/signup/logout
│   ├── patient_routes.py     # Patient views
│   ├── doctor_routes.py      # Doctor views
│   └── component_routes.py   # ML component endpoints
├── templates/                # Jinja2 HTML templates
├── static/                   # CSS + JS
│   ├── css/app.css
│   └── js/firebase_auth.js
├── firebase/                 # Firestore rules & indexes
│   ├── firestore.rules
│   └── firestore.indexes.json
├── CardioSense_Component1/   # Pre-trained Component 1 model
├── CardioSense_Component2/   # Pre-trained Component 2 model
└── CardioSense_Component3/   # Pre-trained Component 3 models
```

## Model Files

Models are referenced from their original component directories:

- **Component 1:** `CardioSense_Component1/cardiosense_component1_model.pkl`
- **Component 2:** `CardioSense_Component2/cardiosense_cxr_densenet121.pth`
- **Component 3:** `CardioSense_Component3/models/` (XGBoost models + meta.json)

## Security

- Firebase Authentication with email/password
- Role-based access via custom claims (`patient` / `doctor`)
- Firestore security rules enforce data isolation
- Server-side session management with Flask
- CSRF protection via session-based tokens
- File upload validation (type + size limits)

## Disclaimer

This is a research prototype. It does not provide medical diagnoses and should not replace professional medical advice.
