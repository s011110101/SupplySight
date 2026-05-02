import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from services.product_config import PRODUCTS, ProductConfig

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "database" / "raw" / "price_data" / "fao_fishpriceindex"
PROCESSED_DIR = ROOT / "database" / "processed"
DEFAULT_FAO_DIR_URL = "https://www.fao.org/fishery/static/fishpriceindex/"
OUTPUT_COLUMNS = ["date", "commodity", "value", "source", "source_file", "ingested_at"]
MONTH_LOOKUP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


class FAOError(RuntimeError):
    pass


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def fetch_directory_listing(dir_url: str, timeout: int = 30) -> str:
    try:
        response = requests.get(dir_url, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise FAOError(f"Failed to fetch directory listing from {dir_url}: {e}")


def _parse_last_modified(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None

    cleaned = " ".join(value.split())
    for fmt in (
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%d-%b-%Y %H:%M",
        "%d-%b-%Y %H:%M:%S",
        "%d %b %Y %H:%M",
        "%d %b %Y %H:%M:%S",
    ):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def _parse_filename_period(filename: str) -> Optional[tuple[int, int]]:
    match = re.search(r"([A-Za-z]{3})[-_ ]?(20\d{2}|\d{2})", filename)
    if not match:
        return None

    month = MONTH_LOOKUP.get(match.group(1).lower())
    if month is None:
        return None

    year_str = match.group(2)
    year = int(year_str)
    if len(year_str) == 2:
        year += 2000 if year < 70 else 1900
    return (year, month)


def parse_csv_links(html: str, dir_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for anchor in soup.find_all("a"):
        href = anchor.get("href", "")
        if ".csv" not in href.lower():
            continue

        filename = href.split("/")[-1]
        if not filename:
            continue

        last_modified = None
        row = anchor.find_parent("tr")
        if row:
            cells = row.find_all("td")
            if len(cells) >= 3:
                last_modified = cells[-1].get_text(strip=True)

        links.append(
            {
                "name": filename,
                "url": urljoin(dir_url.rstrip("/") + "/", href),
                "last_modified_dt": _parse_last_modified(last_modified),
                "period": _parse_filename_period(filename),
            }
        )

    return links


def find_newest_fao_csv(dir_url: str) -> tuple[str, str]:
    links = parse_csv_links(fetch_directory_listing(dir_url), dir_url)
    if not links:
        raise FAOError(f"No CSV files found in FAO directory: {dir_url}")

    candidates = [
        link for link in links if "fish" in link["name"].lower() and "price" in link["name"].lower()
    ]
    if not candidates:
        raise FAOError(f"No FAO Fish Price Index files found in {dir_url}")

    newest = max(
        candidates,
        key=lambda link: (
            link.get("last_modified_dt") or datetime.min,
            link.get("period") or (0, 0),
            link["name"].lower(),
        ),
    )
    return newest["name"], newest["url"]


def download_fao_csv(url: str, filename: str, timeout: int = 30) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    match = re.search(r"([A-Za-z]{3}\d{4})", filename)
    month_part = match.group(1) if match else "unknown"
    snapshot_path = RAW_DIR / f"FAO_fish_price_index_{month_part}__snapshot_{stamp}.csv"

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        snapshot_path.write_text(response.text, encoding="utf-8")
        return snapshot_path
    except requests.RequestException as e:
        raise FAOError(f"Failed to download FAO CSV from {url}: {e}")


def extract_product_series(csv_path: Path, config: ProductConfig) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skiprows=3)
    if config.fao_column not in df.columns or "Date" not in df.columns:
        raise FAOError(f"{config.fao_column} or Date column not found in {csv_path}")

    product = df[["Date", config.fao_column]].copy()
    product.columns = ["date", "value"]
    product["date"] = pd.to_datetime(product["date"], format="%b-%y", errors="coerce").dt.strftime("%Y-%m")
    product["commodity"] = config.fao_column
    product["source"] = "FAO_FishPriceIndex"
    product["source_file"] = csv_path.name
    product["ingested_at"] = datetime.now(timezone.utc).isoformat()
    return product[OUTPUT_COLUMNS]


def extract_shrimp_series(csv_path: Path) -> pd.DataFrame:
    """Backward-compatible helper used by older tests and callers."""
    shrimp = extract_product_series(csv_path, PRODUCTS["shrimp"]).copy()
    shrimp["date"] = pd.to_datetime(shrimp["date"], format="%Y-%m", errors="coerce").dt.strftime("%Y-%m-%d")
    return shrimp


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"])
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out.reset_index(drop=True)


def output_csv_for_product(config: ProductConfig) -> Path:
    return PROCESSED_DIR / f"fao_{config.name}_price_index.csv"


def run(override_url: Optional[str] = None) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    file_url_override = os.getenv("FAO_FISHPRICEINDEX_FILE_URL", "").strip() or None
    dir_url = os.getenv("FAO_FISHPRICEINDEX_DIR_URL", DEFAULT_FAO_DIR_URL).strip()

    if override_url:
        download_url = override_url
        csv_filename = override_url.split("/")[-1]
    elif file_url_override:
        download_url = file_url_override
        csv_filename = download_url.split("/")[-1]
    else:
        csv_filename, download_url = find_newest_fao_csv(dir_url)

    print(f"Found FAO CSV: {csv_filename}")
    print(f"Download URL: {download_url}")

    raw_path = download_fao_csv(download_url, csv_filename)
    print(f"Saved raw snapshot: {raw_path}")

    products_written: dict[str, str] = {}
    rows_by_product: dict[str, int] = {}
    for config in PRODUCTS.values():
        try:
            df_product_clean = clean(extract_product_series(raw_path, config))
        except FAOError:
            continue

        out_csv = output_csv_for_product(config)
        print(f"Extracted {len(df_product_clean)} {config.name} price records")

        if out_csv.exists():
            df_existing = pd.read_csv(out_csv)
            df_existing["date"] = df_existing["date"].astype(str)
            df_merged = pd.concat([df_existing, df_product_clean], ignore_index=True)
        else:
            df_merged = df_product_clean

        df_merged = df_merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        df_merged = df_merged.reset_index(drop=True)
        atomic_write_csv(df_merged, out_csv)
        print(f"Wrote processed CSV: {out_csv}")

        products_written[config.name] = str(out_csv)
        rows_by_product[config.name] = int(len(df_merged))

    return {
        "csv_downloaded": csv_filename,
        "raw_snapshot": str(raw_path),
        "processed_files": products_written,
        "rows_total_by_product": rows_by_product,
    }


if __name__ == "__main__":
    result = run()
    print("\nIngestion complete:")
    for key, value in result.items():
        print(f"  {key}: {value}")
