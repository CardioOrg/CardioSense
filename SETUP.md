# CardioSense — Local Setup Instructions

## Prerequisites

- **Python 3.11+** — [Download](https://www.python.org/downloads/) (check "Add Python to PATH" during install)

---

## Installation

```bash
# 1. Open terminal in the project folder
cd CardioSenseApp

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 4. Install dependencies
pip install -r requirements.txt
```



## Run

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Test Accounts

- Register a patient: http://localhost:5000/signup/patient
- Register a doctor: http://localhost:5000/signup/doctor


Role	Email	Name	Health Profile
🩺 Patient	johndoe@cardiosense.com	            John Doe	        Test@1234
🩺 Patient	sarah.wilson@cardiosense.com	    Sarah Wilson	    Test@1234
🩺 Patient	mike.chen@cardiosense.com	        Mike Chen	        Test@1234
👨‍⚕️ Doctor	dr.smith@cardiosense.com           Dr. Emily Smith	   Test@1234
👨‍⚕️ Doctor	dr.patel@cardiosense.com	       Dr. Raj Patel	   Test@1234