"""Load the Telco Customer Churn dataset.

The raw CSV keeps its original schema. The only cleaning done here is
coercing ``TotalCharges`` to numeric: the source file stores blank strings
for customers with tenure == 0, which become NaN.
"""
from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco-Customer-Churn.csv"
TARGET = "Churn"


def load_raw(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV and return a DataFrame with numeric TotalCharges."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


if __name__ == "__main__":
    df = load_raw()
    print(f"shape: {df.shape}")
    print(f"target distribution:\n{df[TARGET].value_counts(normalize=True).round(4)}")
    print(f"missing values:\n{df.isna().sum()[df.isna().sum() > 0]}")
