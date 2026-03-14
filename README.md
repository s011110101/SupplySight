# SupplySight
## How to Run & Test
1. Setup Environment   
    pip install -r requirements.txt <br>
    set your CENSUS_API_KEY in .env
    > touch .env <br>
    > echo "CENSUS_API_KEY=**your_key_here**\nEIA_API_KEY=**your_key_here**\nSHRIMP_MONTHS_BACK=**number of months you want**" > .env


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

### Oil price (logistics cost signal)
- **Goal:** Capture fuel and logistics cost pressure affecting shrimp supply chains.
- **What it does:**
  - Calls the EIA (Energy Information Administration) API. 
  - Extracts U.S. Gulf Coast Ultra-Low Sulfur No 2 Diesel Spot Price. 
  - Produces a daily dataset stored at database/processed/oil_price_daily.csv.
- **Why it matters:** Diesel prices influence shipping, trucking, and cold-chain logistics costs, which affect supply chain risk.

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

4. oil_price_daily.csv
    > https://www.eia.gov/opendata/
    
    - DATE: <YYYY>-<MM>-<DD>
    - oil_price: U.S. Gulf Coast Ultra-Low Sulfur No 2 Diesel Spot Price (float)

5. news
    tbc

