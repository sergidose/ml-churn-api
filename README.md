# ML Churn API (Telco)

End-to-end churn prediction: train a scikit-learn pipeline and serve predictions with FastAPI.

## Features
- ✅ Download dataset script
- ✅ Training pipeline (OneHot + scaling + LogisticRegression)
- ✅ Metrics saved to `models/metrics.json`
- ✅ FastAPI endpoints: `/health`, `/model-info`, `/predict`
- ✅ Tests with pytest
- ✅ Code quality: Ruff + pre-commit


## Setup (Windows)
```bat
py -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install fastapi uvicorn pandas numpy scikit-learn joblib requests
.venv\Scripts\python -m pip install pytest httpx ruff pre-commit
.venv\Scripts\pre-commit install
```

## Download data
```bat
.venv\Scripts\python scripts\download_data.py
```

## Train
```bat
.venv\Scripts\python -m src.train
```

## Run API
```bat
.venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
```

Open:
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/model-info

## Example request
```bat
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"gender\":\"Female\",\"SeniorCitizen\":0,\"Partner\":\"Yes\",\"Dependents\":\"No\",\"tenure\":12,\"PhoneService\":\"Yes\",\"MultipleLines\":\"No\",\"InternetService\":\"Fiber optic\",\"OnlineSecurity\":\"No\",\"OnlineBackup\":\"Yes\",\"DeviceProtection\":\"No\",\"TechSupport\":\"No\",\"StreamingTV\":\"Yes\",\"StreamingMovies\":\"No\",\"Contract\":\"Month-to-month\",\"PaperlessBilling\":\"Yes\",\"PaymentMethod\":\"Electronic check\",\"MonthlyCharges\":79.85,\"TotalCharges\":950.2}"
```

## Tests
```bat
.venv\Scripts\python -m pytest -q
```

## Run with Docker
```bash
docker compose up --build
```
