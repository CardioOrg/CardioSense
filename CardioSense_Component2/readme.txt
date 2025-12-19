CardioSense Chest X-ray Inference UI
====================================

This repo contains a Streamlit frontend (`app.py`) that serves the trained DenseNet121 multi-label classifier for cardiothoracic findings.

Prerequisites
-------------
- Python 3.10+ recommended
- Model weights file: `cardiosense_cxr_densenet121.pth` in the same folder as `app.py`
- Label map file: `label_map.json` in the same folder (already present)

Setup
-----
1) Create/activate a virtual environment (optional but recommended).
2) Install dependencies:
   pip install -r requirements.txt

Run
---
Launch Streamlit:
   streamlit run app.py

Then open the provided local URL in your browser, upload a PNG/JPG chest X-ray, and adjust the sidebar threshold to see positive calls.

Notes
-----
- Uses GPU if available; otherwise runs on CPU.
- Preprocessing matches training: resize to 224x224, normalize with ImageNet stats.

