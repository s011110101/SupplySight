#!/usr/bin/env python3

from pathlib import Path
import os
from services.oil.client import fetch_oil_price


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "database" / "processed"


def main():

    api_key = os.getenv("EIA_API_KEY")

    if not api_key:
        raise RuntimeError("Missing EIA_API_KEY environment variable")

    print("Fetching oil price data...")

    df = fetch_oil_price(api_key)

    PROCESSED.mkdir(parents=True, exist_ok=True)

    output = PROCESSED / "oil_price_daily.csv"

    df.to_csv(output, index=False)

    print(f"Saved {len(df)} rows → {output}")


if __name__ == "__main__":
    main()