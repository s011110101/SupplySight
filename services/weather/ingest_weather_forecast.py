#!/usr/bin/env python3

import requests
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "database" / "processed"

countries = {
    "india": (17.65, 83.35),
    "ecuador": (-2.35, -80.0),
    "indonesia": (-7.18, 112.75),
    "vietnam": (9.17, 104.8),
    "thailand": (13.45, 100.25),
}


def fetch_marine(lat, lon):

    url = (
        "https://marine-api.open-meteo.com/v1/marine"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=sea_surface_temperature,wave_height"
        "&forecast_days=10"
        "&timezone=UTC"
    )

    r = requests.get(url)
    r.raise_for_status()

    data = r.json()["hourly"]

    df = pd.DataFrame({
        "time": data["time"],
        "sea_surface_temp": data["sea_surface_temperature"],
        "wave_height": data["wave_height"]
    })

    df["DATE"] = pd.to_datetime(df["time"]).dt.floor("D")

    marine_daily = (
        df.groupby("DATE")
        .agg({
            "sea_surface_temp": "mean",
            "wave_height": "mean"
        })
        .reset_index()
    )

    return marine_daily


def fetch_weather(lat, lon):

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=wind_speed_10m_max,precipitation_sum"
        "&forecast_days=10"
        "&timezone=UTC"
    )

    r = requests.get(url)
    r.raise_for_status()

    data = r.json()["daily"]

    df = pd.DataFrame({
        "DATE": pd.to_datetime(data["time"]),
        "wind_speed": data["wind_speed_10m_max"],
        "precipitation": data["precipitation_sum"]
    })

    return df


def fetch_country(country, lat, lon):

    print(f"Fetching forecast for {country}...")

    marine = fetch_marine(lat, lon)
    weather = fetch_weather(lat, lon)

    df = marine.merge(weather, on="DATE", how="outer")

    df = df.rename(columns={
        "sea_surface_temp": f"sea_surface_temp_{country}",
        "wave_height": f"wave_height_{country}",
        "wind_speed": f"wind_speed_{country}",
        "precipitation": f"precipitation_{country}"
    })

    return df


def main():

    dfs = []

    for country, (lat, lon) in countries.items():

        df = fetch_country(country, lat, lon)
        dfs.append(df)

    final = dfs[0]

    for df in dfs[1:]:
        final = final.merge(df, on="DATE", how="outer")

    final = final.sort_values("DATE")

    PROCESSED.mkdir(parents=True, exist_ok=True)

    output = PROCESSED / "daily_forecast_features.csv"

    final.to_csv(output, index=False)

    print(f"\nSaved {len(final)} rows → {output}")


if __name__ == "__main__":
    main()