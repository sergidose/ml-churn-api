from pathlib import Path

import joblib
import pandas as pd

from src.features import ALL_FEATURES


def load_model(path: str | Path):
    return joblib.load(path)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["tenure"] = df["tenure"].astype(int)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    return df


def predict_one(model, payload: dict) -> tuple[float, int]:
    row = {k: payload[k] for k in ALL_FEATURES}
    df = _coerce_types(pd.DataFrame([row]))

    proba = float(model.predict_proba(df)[0, 1])
    pred = int(proba >= 0.5)
    return proba, pred
