# Screening AI — ESS Burn-In Intelligence Platform

An AI-driven prototype for early screening of ESS components using electrical
and thermal drift observed during burn-in testing.

## System architecture

```text
ESS / Synthetic Test Data
        |
        v
data_generator.py
        |
        v
Feature Engineering (0h -> 24h drift)
        |
        +----------------------+
        |                      |
        v                      v
Isolation Forest          XGBoost
Anomaly Detection         168h Failure Prediction
        |                      |
        +----------+-----------+
                   |
                   v
             Risk Decision
        PASS / WARNING / REJECT
                   |
                   v
              SHAP Explanation
                   |
          +--------+--------+
          |                 |
          v                 v
     Streamlit UI       FastAPI API
        app.py          backend/
```

## Project structure

```text
SIH-burning-Screening/
├── app.py                              # Main Streamlit application
├── models.py                           # Shared ML + SHAP engine
├── data_generator.py                   # Synthetic ESS data generator
├── ESS_predictive_screening_dataset_1500.csv
├── requirements.txt                    # Single dependency file
├── README.md
├── .gitignore
└── backend/
    ├── __init__.py
    ├── backend.py                      # FastAPI application
    ├── data_service.py                 # Dataset validation/features
    ├── model_service.py                # API service wrapper
    └── README.md
```

## Run the main application

From the project root:

```bash
python -m streamlit run app.py
```

The main UI does not require the FastAPI server.

## Run the optional API

In a second terminal:

```bash
python -m uvicorn backend.backend:app --reload --port 8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## Generate a fresh dataset

```bash
python data_generator.py
```

This overwrites the CSV in the current working directory.

## Important engineering fix

Dashboard-wide decisions use `ScreeningEngine.predict_batch()` so the app does
not calculate SHAP for every component on every page load. SHAP is calculated
only when an individual component is opened in **AI Insights**.

## Prototype limitation

The included dataset and 168h outcomes are synthetic. This prototype must not
be used for real manufacturing acceptance decisions without validation against
real ESS/burn-in measurements, calibrated thresholds, and engineering approval.
