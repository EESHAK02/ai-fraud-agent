# src/feature_utils.py
import pandas as pd
import numpy as np

def add_basic_features_to_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["user_id","timestamp"]).reset_index(drop=True)

    # time since last txn per user
    prev = df.groupby("user_id")["timestamp"].shift(1)
    df["time_since_last_txn_user"] = (df["timestamp"] - prev).dt.total_seconds().fillna(999999.0)

    # user amount stats (prior-only)
    mean_prior = df.groupby("user_id")["amount"].transform(lambda s: s.expanding().mean().shift(1))
    std_prior  = df.groupby("user_id")["amount"].transform(lambda s: s.expanding().std().shift(1))
    gmean = df["amount"].mean(); gstd = df["amount"].std()
    std_prior = std_prior.replace(0, np.nan).fillna(gstd)
    df["user_amount_mean"] = mean_prior.fillna(gmean)
    df["user_amount_std"]  = std_prior
    df["z_amount_user"]    = (df["amount"] - df["user_amount_mean"]) / df["user_amount_std"]

    # calendar
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    # txns in last 1h per user (safe)
    def _last1h(g):
        g = g.sort_values("timestamp")
        s = pd.Series(1, index=g["timestamp"])
        out = s.rolling("3600s").sum().shift(1)
        out = out.reindex(g["timestamp"]).fillna(0).astype(int)
        out.index = g.index
        return out
    df["txns_last_1h_user"] = df.groupby("user_id", group_keys=False).apply(_last1h, include_groups=False).astype(int)

    # fill numeric only
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df
