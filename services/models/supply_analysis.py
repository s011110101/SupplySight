# pip install openai pandas
import os
import json
import pandas as pd
from openai import OpenAI

USDA_PATH = "database/label_cups_value.csv"          # change to your file
DEMAND_PATH = "database/demand/demand.csv"    # optional; can remove

MODEL = "gpt-4.1"  # or another model you use
MAX_ROWS = 200     # keep small to avoid huge token usage

def load_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".json"):
        return pd.read_json(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")

def df_to_records(df: pd.DataFrame, max_rows: int) -> list:
    if df.empty:
        return []
    # keep it robust even if you don't know types yet
    sample = df.tail(max_rows).copy()
    sample = sample.where(pd.notnull(sample), None)  # NaN -> None
    return sample.to_dict(orient="records")

def main():
    usda_df = load_table(USDA_PATH)
    demand_df = load_table(DEMAND_PATH)

    payload = {
        "usda_sample": df_to_records(usda_df, MAX_ROWS),
        "usda_columns": list(usda_df.columns),
        "demand_sample": df_to_records(demand_df, MAX_ROWS) if not demand_df.empty else [],
        "demand_columns": list(demand_df.columns) if not demand_df.empty else [],
        "demand_context_text": "We expect higher demand for lettuce over the next 60 days.",
    }

    instructions = (
        "You are a supply-chain risk analyst. Using the USDA/open-data sample plus demand context, "
        "estimate supply availability risk for 30/60/90 days and give purchasing guidance. "
        "Be explicit about uncertainty and data gaps. Return bullet points + a short JSON summary."
    )

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.responses.create(
        model=MODEL,
        instructions=instructions,
        input=json.dumps(payload)[:20000],
        max_output_tokens=1200,
    )

    print(resp.output_text)

if __name__ == "__main__":
    main()