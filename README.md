# CardioSense

CardioSense is a 4-part machine learning system for cardiovascular health support:

1. Symptom Analysis & CVD Risk Prediction  
2. Cardiovascular Image Analysis (Chest X-ray)  
3. Personalized Cardiovascular Health Recommendations  
4. Mental Health Chatbot for CVD Patients  

This repo contains working training notebooks for each component, plus Streamlit apps for live demos.

---

## What is completed so far

### Component 1: Symptom Analysis & CVD Risk Prediction
**Notebook:** `CardioSense_Component1_Symptom_Risk_Prediction.ipynb`

What it does:
- Uses a mixed feature set:
  - Structured numeric features (scaled)
  - A text field `symptom_text` vectorized with TF-IDF
- Trains a Logistic Regression classifier
- Saves a single packaged model file: scaler + TF-IDF + classifier

Implementation notes:
- The notebook reads `structured_dataset.csv`, splits train/test, scales numeric features, TF-IDF vectorizes text, then stacks both feature types into one sparse matrix before training. 
- The notebook saves the trained package as `cardiosense_component1_model.pkl`. 

Dataset URL (base source you used elsewhere in the project):
- Kaggle Cardiovascular Disease dataset: :contentReference[oaicite:2]{index=2}  
  (You can build `structured_dataset.csv` by enriching this dataset with generated symptom text.)

---

### Component 2: Cardiovascular Image Analysis (Chest X-ray)
**Notebook:** `CardioSense_Component2_CXR_Full_Colab.ipynb`

What it does:
- Downloads ChestX-ray14 through KaggleHub in Colab
- Builds multi-label targets from the dataset metadata
- Fine-tunes **DenseNet121** with a multi-label head
- Handles class imbalance using `pos_weight` in `BCEWithLogitsLoss`
- Tracks per-label AUC using `torchmetrics`
- Saves the best model weights + label map

Implementation notes:
- Uses pretrained DenseNet121, replaces the classifier, applies `pos_weight`, trains with AdamW, and measures multi-label AUROC. 
- Saves best checkpoint to `outputs/cardiosense_cxr_densenet121.pth` and `outputs/label_map.json`. 

Dataset URL:
- Kaggle NIH Chest X-rays (ChestX-ray14)

---

### Component 3: Personalized Cardiovascular Health Recommendations
**Notebook:** `CardioSense_Component3.ipynb`  
**Demo app:** `app.py` (recommendations + risk scoring)

What it does:
- Trains a **CVD risk model** on structured patient data using XGBoost
- Uses a clean split strategy:
  - Train/test
  - A calibration split
  - An early-stopping validation split
- Calibrates probabilities for more reliable risk scores
- Builds **multi-label recommendation targets** from rule logic
- Trains a **multi-label recommender** (One-vs-Rest)
- Trains a bootstrap ensemble for confidence estimation
- Exports models + cleaned datasets for real-world use

Implementation notes:
- XGBoost training uses early stopping via constructor `early_stopping_rounds` (XGBoost 3.1.x change). 
- Saves exported artifacts to `/content/cardiosense_component3/` and writes model files + `cardio_cleaned.csv` + `recommendation_labels.csv`. 
- The demo app loads the calibrated risk model and bootstrap recommender models, then produces a risk band + prioritized recommendations.

Dataset :
- Kaggle Cardiovascular Disease dataset (70,000 records)
---

### Component 4: Mental Health Chatbot for CVD Patients
**Notebook:** `CardioSense_Component4_LLM_Chatbot_QLoRA.ipynb`  
**Demo app:** `streamlit_app.py`

What it does (notebook):
- Fine-tunes a TinyLLAMA chat model using **QLoRA (4-bit)** + PEFT LoRA
- Prepares training text from multiple lightweight conversation/QA datasets
- Saves:
  - LoRA adapter
  - Optional merged model folder for local inference

Implementation notes:
- Uses 4-bit quantization (NF4) and LoRA on attention/MLP projection modules. 
- Trains with TRL `SFTTrainer` and saves adapter + merged model outputs. 

What it does (Streamlit app):
- Uses a fixed system prompt for supportive tone and safety routing
- Routes urgent content:
  - self-harm
  - emergency physical symptoms  
  before calling the model

Implementation notes:
- It includes regex routing for self-harm and emergency symptoms.

Dataset URLs used for training:
- MentalChat16K (Hugging Face)
- Empathetic Dialogues LLM format (Hugging Face)
- MedQuAD (Hugging Face)
- CounselChat (Kaggle)

---

## Tech stack

Training:
- Python, NumPy, Pandas
- scikit-learn (TF-IDF, Logistic Regression, calibration, multi-label)
- XGBoost (risk prediction)
- PyTorch + TorchVision (DenseNet121 fine-tuning)
- torchmetrics (multi-label AUROC)
- Hugging Face Transformers + Datasets + TRL + PEFT + bitsandbytes (QLoRA)

Deployment:
- Streamlit (apps)

Data access:
- KaggleHub / Kaggle API (Colab dataset pulls)
- Hugging Face datasets (chat training sets)

---