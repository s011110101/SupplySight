# services/census/client.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import requests
import pandas as pd

BASE = "https://api.census.gov/data/timeseries/intltrade/imports/hs"

class CensusError(RuntimeError):
    pass

@dataclass(frozen=True)
class HSImportsQuery:
    hs_code: str                 # "0306170000" or "030617"
    time_from: str               # "YYYY-MM"
    time_to: str                 # "YYYY-MM"
    fields: List[str]            # e.g., ["I_COMMODITY","GEN_VAL_MO","GEN_QY1_MO"]

def fetch_hs_imports(query: HSImportsQuery, api_key: str, timeout: int = 30) -> pd.DataFrame:
    if not api_key:
        raise ValueError("Missing Census API key. Provide it via env var CENSUS_API_KEY.")

    # Build params with individual YEAR and MONTH parameters
    params: Dict[str, Any] = {
        "get": ",".join(query.fields),
        "I_COMMODITY": query.hs_code,
        "key": api_key,
    }

    # Parse time_from and time_to (YYYY-MM format)
    from_year, from_month = map(int, query.time_from.split("-"))
    to_year, to_month = map(int, query.time_to.split("-"))

    # Build list of (year, month) tuples
    years_months = []
    current_year, current_month = from_year, from_month
    while (current_year, current_month) <= (to_year, to_month):
        years_months.append((current_year, current_month))
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    # Add YEAR and MONTH params as lists (requests will repeat them)
    years = [str(ym[0]) for ym in years_months]
    months = [f"{ym[1]:02d}" for ym in years_months]
    params["YEAR"] = years
    params["MONTH"] = months
    
    # Use params with list values; requests will repeat them
    r = requests.get(BASE, params=params, timeout=timeout)
    if r.status_code not in (200, 204):
        raise CensusError(f"HTTP {r.status_code}: {r.text[:500]}")
    
    # 204 = No Content (no data for this HS code)
    if r.status_code == 204:
        return pd.DataFrame()

    data = r.json()
    if not isinstance(data, list) or len(data) < 2:
        raise CensusError(f"Unexpected response format or no rows returned. Response: {str(data)[:200]}")

    header = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=header)
    
    # Remove duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    # Combine YEAR and MONTH into a single MONTH column (YYYY-MM format)
    if "YEAR" in df.columns and "MONTH" in df.columns:
        df["MONTH"] = df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str).str.zfill(2)
        df = df.drop(columns=["YEAR"])  # Drop original YEAR column after combining
    
    return df