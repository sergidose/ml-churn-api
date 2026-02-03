import os
import subprocess
import sys
from pathlib import Path

import uvicorn


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    sys.path.insert(0, str(root))
    os.chdir(root)

    model_path = root / "models" / "churn_pipeline.joblib"
    data_path = root / "data" / "raw" / "telco_churn.csv"

    # Si no hay modelo, descargamos datos y entrenamos (1ª vez en Docker)
    if not model_path.exists():
        # Si no hay datos, descarga
        if not data_path.exists():
            subprocess.check_call([sys.executable, str(root / "scripts" / "download_data.py")])

        subprocess.check_call([sys.executable, "-m", "src.train"])

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
