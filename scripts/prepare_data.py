"""
Clean and enrich the raw Tetouan City dataset:
- rename columns to snake_case
- parse and sort by datetime
- drop duplicate timestamps
- add derived time features
- write processed CSV + a small sample CSV for the repo

Usage:
    python scripts/prepare_data.py
"""

from pathlib import Path

import pandas as pd

RAW_INPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "tetouan_raw.csv"
PROCESSED_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "tetouan_clean.csv"
SAMPLE_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "tetouan_sample.csv"

SAMPLE_ROWS = 1000

RENAME_MAP = {
    "DateTime": "datetime",
    "Temperature": "temperature",
    "Humidity": "humidity",
    "Wind Speed": "wind_speed",
    "general diffuse flows": "general_diffuse_flows",
    "diffuse flows": "diffuse_flows",
    "Zone 1 Power Consumption": "zone_1_consumption",
    "Zone 2 Power Consumption": "zone_2_consumption",
    "Zone 3 Power Consumption": "zone_3_consumption",
}


def main():
    df = pd.read_csv(RAW_INPUT_PATH)
    df.columns = [" ".join(col.split()) for col in df.columns]
    df = df.rename(columns=RENAME_MAP)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    n_duplicates = df["datetime"].duplicated().sum()
    print(f"Duplicate timestamps found: {n_duplicates}")

    print("Null values per column:")
    print(df.isnull().sum().to_string())

    if n_duplicates > 0:
        before = len(df)
        df = df.drop_duplicates(subset="datetime", keep="first")
        after = len(df)
        print(f"WARNING: dropped {before - after} duplicate-timestamp rows.")

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["date"] = df["datetime"].dt.date

    PROCESSED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_OUTPUT_PATH, index=False)

    sample_df = df.head(SAMPLE_ROWS)
    sample_df.to_csv(SAMPLE_OUTPUT_PATH, index=False)

    processed_size_mb = PROCESSED_OUTPUT_PATH.stat().st_size / (1024 * 1024)
    sample_size_mb = SAMPLE_OUTPUT_PATH.stat().st_size / (1024 * 1024)

    print("=== Summary ===")
    print(f"Final row count: {len(df)}")
    print(f"Datetime range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Final columns: {list(df.columns)}")
    print(f"Processed file size: {processed_size_mb:.2f} MB ({PROCESSED_OUTPUT_PATH})")
    print(f"Sample file size: {sample_size_mb:.2f} MB ({SAMPLE_OUTPUT_PATH})")


if __name__ == "__main__":
    main()
