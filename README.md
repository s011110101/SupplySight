# SupplySight

## Requirements

- Python 3.10 or higher
- Node.js 18 or higher, npm 9 or higher
## Setup

### 1. Create your `.env` file (repo root)

The database is hosted on Supabase — credentials were shared by Chenyue in the group chat. Ask Valentina for the `CENSUS_API_KEY`.

```
PGHOST=aws-0-us-west-2.pooler.supabase.com
PGPORT=5432
POSTGRES_USER=postgres.zgtpholoddoruumqtfeu
POSTGRES_PASSWORD=<see group chat>
POSTGRES_DB=supplysight
CENSUS_API_KEY=<ask Valentina>
```

No Docker or local database setup needed — the DB is already running in the cloud.

### 2. Get the ML model files

The trained model files are **not stored in git** (they're large binary artifacts). Ask **Vasili** to send you the following two files and place them in the `models/` folder at the repo root:

```
models/supply_risk_regression.joblib
models/supply_risk_classifier.joblib
```

### 3. Create and activate a virtual environment

```bash
cd SupplySight
python3 -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
```

Always make sure the virtual environment is **active** (you should see `(.venv)` in your prompt) before running any Python commands.

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the FastAPI backend (run from repo root)

```bash
python -m uvicorn services.api.main:app --reload --reload-dir services --port 8000
```

Keep this terminal open — the frontend proxies all `/api` requests to it.

### 5. Install frontend dependencies and start the dev server

```bash
cd Dashboard
npm install
npm run dev
```

Open **http://localhost:3000** in your browser.

## High-level overview of each data pipeline

### Shrimp imports (Census)
- **Goal:** Build a monthly history of US shrimp import volumes by product.  
- **What it does:**
  - Calls the US Census trade API for shrimp-related HS codes.
  - Collects monthly values and weights (total, vessel, container, air) plus product descriptions.
  - Cleans the data, removes duplicates, and appends new months to the existing history.
  - Writes a timestamped raw snapshot to `database/raw/shrimp_imports/` and an up-to-date processed file `database/processed/shrimp_imports.csv`.
- **Why it matters:** This is the core **supply signal** (how much shrimp actually arrived) that everything else is built around.

### FAO shrimp price index
- **Goal:** Add a standardized price signal that reflects the global demand–supply balance for shrimp.  
- **What it does:**
  - Downloads the FAO shrimp price index from a configured source URL.
  - Normalizes dates and keeps key metadata (commodity name, source, source file, ingestion timestamp).
  - Produces `database/processed/fao_shrimp_price_index.csv`.
- **Why it matters:** Prices help explain and predict supply changes; they summarize tight vs loose market conditions.

### Ocean weather
- **Goal:** Capture environmental conditions that affect shrimp production and catch.  
- **What it does:**
  - Calls the Open-Meteo API for ocean variables such as sea-surface temperature, wave height, currents, and sea level.
  - Stores hourly data in `database/processed/weather_hourly.csv`.
  - Aggregates that data to monthly features (e.g. average temperature, max wave height) in `database/processed/weather_features.csv`.
- **Why it matters:** Ocean and weather conditions are leading indicators for future shrimp availability.

### News (market and disruption signals)
- **Goal:** Track qualitative signals about shrimp markets, disruptions, and sentiment.  
- **What it does:**
  - Uses NewsAPI to fetch recent shrimp-related articles.
  - Stores raw articles in Postgres for durability and recovery.
  - Uses Claude (Anthropic) to transform raw text into structured summaries and tags, then loads them back into Postgres.
- **Why it matters:** News can capture shocks and structural changes (policy, disease, logistics) that don’t immediately appear in prices or volumes but affect future supply risk.



2. run data collection
  

## dataframe
1. shrimp_imports.cvs:
    >https://www.census.gov/data/developers/data-sets/international-trade.html<br>
    https://www.census.gov/foreign-trade/reference/guides/Guide_to_International_Trade_Datasets.pdf
    
    - I_COMMODITY: 2, 4, 6, or 10 character Import Harmonized System Code (String)
    - I_COMMODITY_SDESC: 50 character Import Harmonized Code Description (String)
    - GEN_VAL_MO: General Imports, Total Value (Int)
    - VES_WGT_MO: Vessel Shipping Weight (Int)
    - CNT_WGT_MO: Containerized Vessel Shipping Weight (Int)
    - AIR_WGT_MO: Air Shipping Weight (Int)
    - MONTH: \<YYYY\>\-\<MM\>
2. fao_shrimp_price_index.csv
    > https://www.fao.org/fishery/en/fishstat/fishpriceindex
    
    - date: \<YYYY\>\-\<MM\>
    - commodity: (String)
    - value: (float .1f)
    - source: (String)
    - source_file: (String)
    - ingested_at: (String)
3. weather_hourly.csv
    > https://open-meteo.com/en/docs
    
    - time (YYYY-MM-DDThh:mm)
    - sea_surface_temperature: (float .1f)
    - wave_height (float .2f)
    - ocean_current_velocity (float .1f)
    - sea_level_height_msl (float .2f)
4. news
    tbc

It should look something like this
<img width="1360" height="782" alt="Screen Shot 2026-04-30 at 11 32 51 PM" src="https://github.com/user-attachments/assets/5ff65b84-7a2a-4887-aa94-67c7cc709121" />

