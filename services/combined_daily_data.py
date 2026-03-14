#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "database" / "processed"

WEATHER = PROCESSED / "weather_daily_historical.csv"
OUTPUT = PROCESSED / "daily_training_data.csv"


def load_weather():

    if not WEATHER.exists():
        raise FileNotFoundError(f"Missing weather dataset: {WEATHER}")

    df = pd.read_csv(WEATHER)

    df["DATE"] = pd.to_datetime(df["DATE"])

    return df


def main():

    print("Building daily training dataset...")

    weather = load_weather()

    combined = weather.sort_values("DATE").reset_index(drop=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)

    combined.to_csv(OUTPUT, index=False)

    print(f"Wrote {len(combined)} rows → {OUTPUT}")
    print("Columns:", list(combined.columns))


if __name__ == "__main__":
    main()