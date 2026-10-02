# Phase 3 Frontend

This folder contains the Streamlit frontend for the AI Burn-In Screening project.

## Folder
Place this folder inside:

C:\Users\legen\Downloads\sih_burnin_screening\sih_burnin_screening\

So the path becomes:

sih_burnin_screening/
├── phase2/
├── models.py
├── data_generator.py
├── app.py
└── phase3_frontend/
    ├── app.py
    ├── requirements.txt
    └── README.md

## Install frontend packages

From the project root:

python -m pip install -r phase3_frontend\requirements.txt

## Run Phase 2 backend

Keep one PowerShell window running:

python -m uvicorn phase2.backend:app --reload --port 8000

## Run Phase 3 frontend

Open a second PowerShell window in the same project root:

python -m streamlit run phase3_frontend\app.py

The frontend normally opens at:

http://localhost:8501

The backend is:

http://127.0.0.1:8000

Swagger:

http://127.0.0.1:8000/docs
