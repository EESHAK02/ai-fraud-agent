# src/data_loader.py
from pathlib import Path
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# CONFIG: how many synthetic users/merchants to create
N_SYN_USERS = 10000      # tweak for realism
N_SYN_MERCHANTS = 2000

def load_creditcard_csv(path: Path = None,
                        n_users: int = N_SYN_USERS,
                        n_merchants: int = N_SYN_MERCHANTS,
                        time_base: str = "2013-01-01T00:00:00"):
    """
    Load Kaggle creditcard.csv and create canonical columns.
    - path: Path to creditcard.csv
    - returns: pandas DataFrame with canonical columns + V1..V28 kept
    """
    if path is None:
        path = RAW_DIR / "creditcard.csv"
    df = pd.read_csv(path)

    # Ensure expected columns present
    expected = {"Time", "Amount", "Class"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"CSV missing expected columns: {expected - set(df.columns)}")

    # create a synthetic absolute timestamp from Time (seconds) using a base
    base = pd.to_datetime(time_base)
    df["timestamp"] = pd.to_timedelta(df["Time"], unit="s") + base

    # Keep PCA features (V1..V28) if present
    v_cols = [c for c in df.columns if c.startswith("V")]
    # create unique txn_id
    df["txn_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    # create synthetic user_id and merchant_id by hashing the row index
    rng = np.random.default_rng(42)  # reproducible
    # Option A: assign users uniformly by index modulo
    df["user_id"] = (df.index % n_users).astype(str)

    # Optionally, you can assign merchants at random
    merchant_pool = [f"m_{i}" for i in range(n_merchants)]
    df["merchant_id"] = rng.choice(merchant_pool, size=len(df))

    # device and channel defaults
    df["device_id"] = "dev_unknown"
    df["channel"] = "card"
    df["country"] = "US"
    df["currency"] = "USD"
    df["lat"] = pd.NA
    df["lon"] = pd.NA
    df["mcc"] = pd.NA

    # canonicalize amount and label
    df["amount"] = df["Amount"].astype(float)
    df["label"] = df["Class"].astype(int)

    # select canonical output columns (plus V1..V28)
    out_cols = [
        "txn_id", "user_id", "merchant_id", "amount", "timestamp",
        "country", "currency", "channel", "device_id", "lat", "lon", "mcc", "label"
    ] + v_cols

    out = df[out_cols].copy()
    # save parquet
    out_path = PROC_DIR / "creditcard_canonical.parquet"
    out.to_parquet(out_path, index=False)
    print("Saved canonicalized dataset to:", out_path)
    return out

if __name__ == "__main__":
    load_creditcard_csv()
