import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import ALL_FEATURES, CATEGORICAL_FEATURES, ID_COL, NUMERIC_FEATURES, TARGET_COL


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # TotalCharges a veces tiene espacios vacíos
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["tenure"] = df["tenure"].astype(int)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    return df


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    return Pipeline([("preprocess", pre), ("model", clf)])


def train_and_save(
    df_raw: pd.DataFrame,
    model_path: Path,
    metrics_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    df = _clean_dataframe(df_raw)

    y = df[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)
    X = df[ALL_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "threshold": 0.5,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "trained_at_utc": datetime.now(UTC).isoformat(),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/raw/telco_churn.csv")
    parser.add_argument("--model_path", default="models/churn_pipeline.joblib")
    parser.add_argument("--metrics_path", default="models/metrics.json")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    metrics = train_and_save(df, Path(args.model_path), Path(args.metrics_path))
    print("✅ Training done. Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
