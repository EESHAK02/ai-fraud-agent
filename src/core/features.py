# src/core/features.py
import numpy as np
import pandas as pd
from ..config import WINDOW_SECONDS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build minimal, readable features for fraud detection.

    Input may contain: ['timestamp' or 'Time', 'Amount', 'user_id', 'merchant_id', 'Class', 'txn_id'].
    Output columns (stable schema):
      ['txn_id','timestamp','user_id','merchant_id','Amount','Class','hour',
       'vel_5m','same_amt_5m','fanout_5m','z_amount']
    """
    d = df.copy()

    # ---- Ensure identifiers & basics exist -----------------------------------
    if "txn_id" not in d.columns:
        d["txn_id"] = d.index.astype(str)

    # timestamp: prefer existing; else derive from Kaggle 'Time'; else monotonic
    if "timestamp" in d.columns:
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    elif "Time" in d.columns:
        d["timestamp"] = pd.to_datetime("2013-01-01") + pd.to_timedelta(d["Time"], unit="s")
    else:
        d["timestamp"] = pd.to_datetime("2013-01-01") + pd.to_timedelta(np.arange(len(d)), unit="s")

    # ids
    if "user_id" not in d.columns:
        d["user_id"] = (d.index % 5000).astype(str)
    if "merchant_id" not in d.columns:
        d["merchant_id"] = ("m_" + (d.index % 2000).astype(str))

    # amount/label
    if "Amount" not in d.columns:
        d["Amount"] = 0.0
    if "Class" not in d.columns:
        d["Class"] = 0

    # clean & sort
    d = d.dropna(subset=["timestamp"]).copy()
    d = d.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # hour-of-day
    d["hour"] = d["timestamp"].dt.hour

    # ---- Rolling-window per user (position-safe) -----------------------------
    def _window_feats(g: pd.DataFrame) -> pd.DataFrame:
        # work on a fresh 0..n-1 index to avoid label-based surprises
        g = g.reset_index(drop=True)

        # epoch seconds as int64 array for fast comparisons
        ts = g["timestamp"].values.astype("datetime64[s]").astype("int64")
        n = len(g)

        vel = np.ones(n, dtype=int)
        same = np.ones(n, dtype=int)
        fan = np.ones(n, dtype=int)

        lo = 0
        for i in range(n):
            t_now = ts[i]
            # slide window start to keep only last WINDOW_SECONDS
            while lo < i and ts[lo] < t_now - WINDOW_SECONDS:
                lo += 1
            win = g.iloc[lo:i + 1]
            vel[i] = len(win)
            same[i] = (win["Amount"] == g.iloc[i]["Amount"]).sum()
            fan[i] = win["merchant_id"].nunique()

        g["vel_5m"] = vel
        g["same_amt_5m"] = same
        g["fanout_5m"] = fan
        return g

    d = d.groupby("user_id", group_keys=False, sort=False).apply(_window_feats)

    # ---- Z-amount per user ---------------------------------------------------
    try:
        stats = (
            d.groupby("user_id", observed=True)["Amount"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "mu", "std": "sd"})
        )
        d = d.join(stats, on="user_id")
        z = (d["Amount"] - d["mu"]) / d["sd"].replace(0, np.nan)
        d["z_amount"] = z.fillna(0.0).clip(-5, 5).abs()
    except Exception:
        d["z_amount"] = 0.0

    # ---- Final column order --------------------------------------------------
    keep = [
        "txn_id",
        "timestamp",
        "user_id",
        "merchant_id",
        "Amount",
        "Class",
        "hour",
        "vel_5m",
        "same_amt_5m",
        "fanout_5m",
        "z_amount",
    ]

    for k in keep:
        if k not in d.columns:
            if k in ("Amount", "z_amount"):
                d[k] = 0.0
            elif k in ("hour", "vel_5m", "same_amt_5m", "fanout_5m"):
                d[k] = 0
            elif k == "Class":
                d[k] = 0
            elif k == "timestamp":
                d[k] = pd.to_datetime("2013-01-01") + pd.to_timedelta(np.arange(len(d)), unit="s")
            elif k == "txn_id":
                d[k] = d.index.astype(str)
            elif k == "user_id":
                d[k] = (d.index % 5000).astype(str)
            elif k == "merchant_id":
                d[k] = ("m_" + (d.index % 2000).astype(str))

    return d[keep]
