from pathlib import Path

import requests

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def main() -> None:
    out = Path("data/raw/telco_churn.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    r = requests.get(URL, timeout=60)
    r.raise_for_status()

    out.write_bytes(r.content)
    print(f"✅ Saved dataset to: {out} ({len(r.content)} bytes)")


if __name__ == "__main__":
    main()
