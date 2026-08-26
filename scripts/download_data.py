"""
Download the "Power Consumption of Tetouan City" dataset (UCI id=849) and
save the raw, untouched combined dataframe to data/raw/tetouan_raw.csv.

Usage:
    python scripts/download_data.py
"""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

RAW_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "tetouan_raw.csv"


def main():
    dataset = fetch_ucirepo(id=849)
    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)

    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_OUTPUT_PATH, index=False)

    print(f"Saved raw dataset to: {RAW_OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Column names: {list(df.columns)}")


if __name__ == "__main__":
    main()
