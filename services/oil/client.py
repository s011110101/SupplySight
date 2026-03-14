from __future__ import annotations
import requests
import pandas as pd

BASE = "https://api.eia.gov/v2/petroleum/pri/spt/data/"

class OilAPIError(RuntimeError):
    pass


def fetch_oil_price(api_key: str, start: str = "2018-01-01") -> pd.DataFrame:

    if not api_key:
        raise ValueError("Missing EIA API key. Set env var EIA_API_KEY.")

    params = {
        "api_key": api_key,
        "frequency": "daily",
        "data[0]": "value",
        "start": start,
    }

    r = requests.get(BASE, params=params, timeout=30)

    if r.status_code != 200:
        raise OilAPIError(f"HTTP {r.status_code}: {r.text[:300]}")

    data = r.json()["response"]["data"]

    df = pd.DataFrame(data)

    df = df[df["series-description"] ==
        "U.S. Gulf Coast Ultra-Low Sulfur No 2 Diesel Spot Price (Dollars per Gallon)"]

    df = df.rename(columns={
        "period": "DATE",
        "value": "oil_price"
    })

    df["DATE"] = pd.to_datetime(df["DATE"])

    return df[["DATE", "oil_price"]].sort_values("DATE")