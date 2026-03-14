#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "database" / "processed"

WEATHER = PROCESSED / "weather_daily_historical.csv"
OIL = PROCESSED / "oil_price_daily.csv"

OUTPUT = PROCESSED / "daily_training_raw_data.csv"


def load_weather():

    if not WEATHER.exists():
        raise FileNotFoundError(f"Missing weather dataset: {WEATHER}")

    df = pd.read_csv(WEATHER)

    df["DATE"] = pd.to_datetime(df["DATE"])

    return df

def load_oil():

    if not OIL.exists():
        raise FileNotFoundError(f"Missing oil dataset: {OIL}")

    df = pd.read_csv(OIL)

    df["DATE"] = pd.to_datetime(df["DATE"])

    return df


def main():

    print("Building daily training dataset...")

    weather = load_weather()
    oil = load_oil()

    combined = weather.merge(oil, on="DATE", how="left")
    combined["oil_price"] = combined["oil_price"].ffill()

    combined = combined.sort_values("DATE").reset_index(drop=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)

    combined.to_csv(OUTPUT, index=False)

    print(f"Wrote {len(combined)} rows → {OUTPUT}")
    print("Columns:", list(combined.columns))


if __name__ == "__main__":
    main()