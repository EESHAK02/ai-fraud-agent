# src/adversary/detectors.py
"""
Rule-based burst detector for the Agentic Fraud Detector.

Computes a per-transaction "rule_burst_score" based on
short-window transaction counts, repeated amounts,
and device fanout patterns.
"""

import pandas as pd
import numpy as np

def rule_burst_score(df: pd.DataFrame, window_seconds: int = 300) -> pd.Series:
    """
    Compute a lightweight, interpretable burstiness score per transaction.
    Higher = more suspicious.
    Parameters
    ----------
    df : pd.DataFrame
        Must include ['timestamp','user_id','amount','device_id'] columns.
    window_seconds : int
        Sliding window size in seconds (default = 5 minutes).
    Returns
    -------
    pd.Series : numeric score aligned to df.index
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- 1. Txns per user in short window ---
    user_counts = []
    for uid, g in df.groupby("user_id"):
        g = g.sort_values("timestamp")
        s = pd.Series(1, index=g["timestamp"])
        c = s.rolling(f"{window_seconds}s").sum().shift(1)
        user_counts.append(pd.Series(c.values, index=g.index))
    user_counts = pd.concat(user_counts).sort_index()
    df["user_txns_last_window"] = user_counts.fillna(0)

    # --- 2. Same-amount frequency globally ---
    df["amount_rounded"] = df["amount"].round(2)
    amt_counts = []
    for amt, g in df.groupby("amount_rounded"):
        g = g.sort_values("timestamp")
        s = pd.Series(1, index=g["timestamp"])
        c = s.rolling(f"{window_seconds}s").sum().shift(1)
        amt_counts.append(pd.Series(c.values, index=g.index))
    amt_counts = pd.concat(amt_counts).sort_index()
    df["same_amt_last_window"] = amt_counts.fillna(0)

    # --- 3. Device fanout: #unique users per device in last 24h ---
    df["device_user_count_24h"] = 0
    for dev, g in df.groupby("device_id"):
        g = g.sort_values("timestamp")
        res = []
        for i, t in enumerate(g["timestamp"]):
            start_t = t - pd.Timedelta(hours=24)
            res.append(g.loc[(g["timestamp"] >= start_t) & (g["timestamp"] < t), "user_id"].nunique())
        df.loc[g.index, "device_user_count_24h"] = res

    # --- 4. Normalize & combine ---
    def norm(s):
        s = s.astype(float)
        if s.max() == s.min():
            return np.zeros(len(s))
        return (s - s.min()) / (s.max() - s.min())

    u_norm = norm(df["user_txns_last_window"])
    a_norm = norm(df["same_amt_last_window"])
    d_norm = norm(df["device_user_count_24h"])

    # weighted sum (tunable weights)
    score = 0.5 * u_norm + 0.35 * a_norm + 0.15 * d_norm
    return pd.Series(score, index=df.index)
