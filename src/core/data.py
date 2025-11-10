# src/core/data.py
import pandas as pd
import numpy as np
from pathlib import Path
from ..config import DATA_PATH, DEMO_MAX_ROWS
from .features import build_features

def load_or_demo():
    p = Path(DATA_PATH)
    if p.exists():
        # Faster, memory-friendly load. Use nrows if you like, or sample afterwards.
        df = pd.read_csv(p)
        # Normalize expected fields
        df["timestamp"] = pd.to_datetime("2013-01-01") + pd.to_timedelta(df["Time"], unit="s")
        df["user_id"] = (df.index % 5000).astype(str)           # simple synthetic users
        df["merchant_id"] = ("m_" + (df.index % 2000).astype(str))
        df["txn_id"] = df.index.astype(str)

        # CAP total rows to keep laptop happy
        if len(df) > DEMO_MAX_ROWS:
            # stratified-ish: keep all frauds + random sample of non-frauds
            fraud = df[df["Class"] == 1]
            non   = df[df["Class"] == 0].sample(
                n=max(1000, DEMO_MAX_ROWS - len(fraud)),
                random_state=42
            )
            df = pd.concat([fraud, non], ignore_index=True).sort_values("Time").reset_index(drop=True)
        return df

    # Fallback tiny synthetic if Kaggle CSV absent
    ts = pd.date_range("2013-01-01", periods=8000, freq="s")
    df = pd.DataFrame({
        "txn_id": ts.astype(int).astype(str),
        "timestamp": ts,
        "user_id": (np.arange(len(ts)) % 1000).astype(str),
        "merchant_id": ("m_" + (np.arange(len(ts)) % 300).astype(str)),
        "Amount": 10 + (np.arange(len(ts)) % 50) * 0.1,
        "Class": 0
    })
    return df

def prepare(df: pd.DataFrame):
    return build_features(df)
