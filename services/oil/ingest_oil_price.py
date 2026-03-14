import requests
import pandas as pd
from pathlib import Path

API_KEY = "bFF7I0Ow8Rc8d9yVPPdJaAAd68nEOSJkHovzP4GJ"

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "database" / "processed"

url = (
    "https://api.eia.gov/v2/petroleum/pri/spt/data/"
    f"?api_key={API_KEY}"
    "&frequency=daily"
    "&data[0]=value"
    "&start=2018-01-01"
)

r = requests.get(url).json()
data = r["response"]["data"]

df = pd.DataFrame(data)

df = df[df["series-description"] ==
        "U.S. Gulf Coast Ultra-Low Sulfur No 2 Diesel Spot Price (Dollars per Gallon)"]

df = df.rename(columns={
    "period": "DATE",
    "value": "oil_price"
})

df["DATE"] = pd.to_datetime(df["DATE"])

df = df[["DATE", "oil_price"]]

df = df.sort_values("DATE")

PROCESSED.mkdir(parents=True, exist_ok=True)

df.to_csv(PROCESSED / "oil_price_daily.csv", index=False)

print("Saved oil_price_daily.csv")