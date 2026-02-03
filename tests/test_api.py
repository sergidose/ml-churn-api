import tempfile
from importlib import reload
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from src.train import train_and_save


def _tiny_df() -> pd.DataFrame:
    rows = []
    for i in range(30):
        rows.append(
            {
                "customerID": f"id-{i}",
                "gender": "Female" if i % 2 == 0 else "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes" if i % 3 == 0 else "No",
                "Dependents": "No",
                "tenure": i,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic" if i % 2 == 0 else "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.0 + (i % 5),
                "TotalCharges": float((70.0 + (i % 5)) * max(i, 1)),
                "Churn": "Yes" if i % 4 == 0 else "No",
            }
        )
    return pd.DataFrame(rows)


def test_health():
    import app.main as main_mod

    client = TestClient(main_mod.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_after_training(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "model.joblib"
        metrics_path = Path(tmp) / "metrics.json"

        train_and_save(_tiny_df(), model_path, metrics_path)

        monkeypatch.setenv("MODEL_PATH", str(model_path))
        monkeypatch.setenv("METRICS_PATH", str(metrics_path))

        import app.main as main_mod

        reload(main_mod)
        client = TestClient(main_mod.app)

        payload = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 79.85,
            "TotalCharges": 950.2,
        }

        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["churn_prediction"] in {"Yes", "No"}
