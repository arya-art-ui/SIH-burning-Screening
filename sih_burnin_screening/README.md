# AI-Driven Component Burn-In & Screening

A synthetic AI prototype for early screening of electronic components using electrical and thermal drift observed during ESS/burn-in testing.

## Features

- 1,500 synthetic components
- 10 manufacturing lots
- 0h, 24h, 96h and 168h ESS measurements
- Voltage, current and temperature
- Early 0h-to-24h drift features
- Isolation Forest anomaly detection
- XGBoost 168h failure prediction
- SHAP explainability
- PASS / WARNING / REJECT decision engine
- Streamlit interactive dashboard
- Lot-level reliability overview
- CSV dataset download

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Generate the dataset

```bash
python data_generator.py
```

## Run the dashboard

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Project flow

ESS Data
    ↓
Feature Engineering
    ↓
Isolation Forest + XGBoost
    ↓
Risk Engine
    ↓
SHAP Explanation
    ↓
Streamlit Dashboard

## Important note

This is a synthetic demonstration dataset and AI prototype. It should not be used for real manufacturing decisions without validation against real ESS/burn-in measurements and appropriate engineering acceptance criteria.
