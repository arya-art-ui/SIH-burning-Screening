# Phase 2 Backend

## Where this folder goes

Put the entire `phase2` folder INSIDE your existing project folder:

C:\Users\legen\Downloads\sih_burnin_screening\sih_burnin_screening\

Final structure:

sih_burnin_screening/
├── app.py
├── data_generator.py
├── models.py
├── ESS_predictive_screening_dataset_1500.csv
└── phase2/
    ├── __init__.py
    ├── backend.py
    ├── data_service.py
    ├── model_service.py
    ├── requirements.txt
    └── README.md

## Install

From the main project folder:

python -m pip install -r phase2\requirements.txt

## Start API

From the main project folder:

python -m uvicorn phase2.backend:app --reload --port 8000

## Test

Health:
http://127.0.0.1:8000/health

API documentation:
http://127.0.0.1:8000/docs

## Important

Keep your existing app.py and models.py unchanged for this step.
