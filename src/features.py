# src/features.py
import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import PROC_DIR


# ROOT = Path(__file__).resolve().parents[1]
# PROC_DIR = ROOT / "data" / "processed"

NUMERIC_DEFAULT = 0.0

def _txns_last_1h_per_user(g: pd.DataFrame) -> pd.Series:
    """
    Safe rolling count of previous 1h transactions per user.
    Uses timestamp as index, counts past 1h, then shifts by 1 to exclude current row.
    """
    g = g.sort_values("timestamp")
    # 1 per transaction
    s = pd.Series(1, index=g["timestamp"])
    # rolling sum over 1 hour, then shift to exclude current txn
    last_1h = s.rolling("3600s").sum().shift(1)
    # align back to original row order
    out = last_1h.reindex(g["timestamp"]).fillna(0).astype(int)
    out.index = g.index  # preserve row alignment to original df
    return out

def add_basic_features(in_path: Path = None, out_name: str = "creditcard_features.parquet"):
    if in_path is None:
        in_path = PROC_DIR / "creditcard_canonical.parquet"

    df = pd.read_parquet(in_path)

    # Ensure timestamp is proper datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Sort for time-aware ops
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # --- Time since last txn per user ---
    df["prev_ts"] = df.groupby("user_id")["timestamp"].shift(1)
    # compute seconds; NaNs (first txn per user) → large sentinel
    tsl = (df["timestamp"] - df["prev_ts"]).dt.total_seconds()
    df["time_since_last_txn_user"] = tsl.fillna(999999.0)

    # --- User amount stats (expanding) using prior history only ---
    user_mean = df.groupby("user_id")["amount"].transform(
        lambda s: s.expanding().mean().shift(1)
    )
    user_std = df.groupby("user_id")["amount"].transform(
        lambda s: s.expanding().std().shift(1)
    )

    global_mean = df["amount"].mean()
    global_std  = df["amount"].std()

    df["user_amount_mean"] = user_mean.fillna(global_mean)
    # avoid zero std → fall back to global std
    df["user_amount_std"]  = user_std.replace(0, np.nan).fillna(global_std)

    df["z_amount_user"] = (df["amount"] - df["user_amount_mean"]) / df["user_amount_std"]

    # --- Calendar features ---
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    # --- Transactions in last 1h per user (safe) ---
    df["txns_last_1h_user"] = (
        df.groupby("user_id", group_keys=False)
        .apply(_txns_last_1h_per_user, include_groups=False)  # <-- add include_groups=False
        .astype(int)
    )

    # Drop temporary datetime helper to avoid parquet dtype issues
    df = df.drop(columns=["prev_ts"], errors="ignore")

    # Fill only numeric NaNs; leave datetime/categorical alone
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(NUMERIC_DEFAULT)

    out_path = PROC_DIR / out_name
    df.to_parquet(out_path, index=False)
    print("Saved features to:", out_path)
    return df

if __name__ == "__main__":
    add_basic_features()
