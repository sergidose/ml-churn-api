import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import load_model, predict_one

BASE_DIR = Path(__file__).resolve().parents[1]  # raíz del repo
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "churn_pipeline.joblib")))
METRICS_PATH = Path(os.getenv("METRICS_PATH", str(BASE_DIR / "models" / "metrics.json")))

app = FastAPI(title="ML Churn API")


_model = None


class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)

    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.on_event("startup")
def _startup():
    global _model
    if MODEL_PATH.is_file():
        _model = load_model(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    if not METRICS_PATH.is_file():
        raise HTTPException(status_code=404, detail=f"metrics.json not found at: {METRICS_PATH}")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(payload: CustomerFeatures):
    global _model

    # ✅ Lazy load: si el modelo no está cargado, intenta cargarlo ahora
    if _model is None:
        if MODEL_PATH.is_file():
            _model = load_model(MODEL_PATH)
        else:
            raise HTTPException(
                status_code=503, detail="Model not loaded. Run: python -m src.train"
            )

    proba, pred = predict_one(_model, payload.model_dump())
    return {
        "churn_probability": proba,
        "churn_prediction": "Yes" if pred == 1 else "No",
        "threshold": 0.5,
    }
