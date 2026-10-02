from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

from phase2.data_service import prepare_dataframe
from phase2.model_service import ScreeningService

app = FastAPI(
    title="AI Burn-In Screening API",
    version="2.0.0",
    description="Backend API for the AI-Driven Component Burn-In & Screening Engine."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = ScreeningService()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_trained": service.is_trained
    }


@app.post("/train")
async def train(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
        df = prepare_dataframe(df)
        service.train(df)

        return {
            "message": "Model trained successfully.",
            "rows": len(df),
            "columns": len(df.columns),
            "accuracy": service.metrics["accuracy"],
            "roc_auc": service.metrics["roc_auc"]
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/predict")
async def predict(payload: dict):
    if not service.is_trained:
        raise HTTPException(
            status_code=400,
            detail="Model is not trained. Call /train first."
        )

    try:
        return service.predict(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
