# src/core/attack.py
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

def _ensure_timestamp_col(df: pd.DataFrame):
    if "timestamp" not in df.columns and "Time" in df.columns:
        df["timestamp"] = pd.to_datetime("2013-01-01") + pd.to_timedelta(df["Time"], unit="s")
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime("2013-01-01") + pd.to_timedelta(np.arange(len(df)), unit="s")
    return df

def inject_burst(
    df: pd.DataFrame,
    n: int = 150,
    amount: float = 49.99,
    interval_s: int = 1,
    merchants_pool: int = 50,
    adv_user: str = "adv_user_0",
    t0: Optional[pd.Timestamp] = None,
    template_frac: float = 0.01,
    mode: str = "clone",   # "clone" | "sample_v" | "perturb_v"
    jitter_time_s: float = 0.0,
    amount_jitter_pct: float = 0.0,
    v_noise_scale: float = 0.0,
    distributed_users: bool = False,
    users_pool_size: int = 1,
    force_label_class: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Inject realistic attack burst rows into `df` while preserving PCA columns V1..V28.

    Returns (df_with_attacks, metadata)
    """

    df = df.copy()
    df = _ensure_timestamp_col(df)
    if t0 is None:
        t0 = df["timestamp"].min() + pd.Timedelta(minutes=10)

    # detect V columns (V1..V28)
    v_cols = sorted([c for c in df.columns if c.startswith("V")])
    has_v = len(v_cols) > 0

    # candidate pool: prefer non-fraud rows
    if "Class" in df.columns:
        pool = df[df["Class"] == 0].copy()
        if len(pool) == 0:
            pool = df.copy()
    else:
        pool = df.copy()

    # sample templates to reduce cost
    templates = pool.sample(n=max(1, int(len(pool) * template_frac)), replace=True, random_state=42)

    merchants = [f"m_adv_{i}" for i in range(merchants_pool)]
    rows = []
    for i in range(n):
        tpl = templates.sample(n=1, random_state=(42 + i)).iloc[0].to_dict()

        # mode handling
        if mode == "sample_v" and has_v:
            sample_row = pool.sample(n=1, random_state=(1000 + i)).iloc[0]
            for c in v_cols:
                tpl[c] = sample_row[c]
        if mode == "perturb_v" and has_v and v_noise_scale > 0:
            for c in v_cols:
                tpl[c] = float(tpl.get(c, 0.0) + np.random.normal(0, v_noise_scale))

        tpl["txn_id"] = f"adv_{i}"
        jitter = int(np.round(np.random.normal(0, jitter_time_s))) if jitter_time_s > 0 else 0
        tpl["timestamp"] = (t0 + pd.Timedelta(seconds=int(i * interval_s + jitter)))
        if distributed_users and users_pool_size > 1:
            tpl["user_id"] = f"{adv_user}_{i % users_pool_size}"
        else:
            tpl["user_id"] = adv_user
        tpl["merchant_id"] = merchants[i % len(merchants)]
        if amount_jitter_pct > 0:
            amt = float(amount * (1.0 + np.random.normal(0, amount_jitter_pct)))
        else:
            amt = float(amount)
        tpl["Amount"] = round(amt, 2)
        if force_label_class is not None:
            tpl["Class"] = force_label_class
        tpl["is_attack"] = 1
        rows.append(tpl)

    attacks_df = pd.DataFrame(rows)
    # Ensure same columns exist; keep original columns order
    for c in df.columns:
        if c not in attacks_df.columns:
            attacks_df[c] = np.nan

    if "timestamp" in attacks_df.columns:
        attacks_df["timestamp"] = pd.to_datetime(attacks_df["timestamp"])

    df_out = pd.concat([df, attacks_df[df.columns.union(["is_attack"])]], ignore_index=True, sort=False)
    meta = {
        "n_injected": n,
        "t0": t0,
        "merchants": merchants,
        "mode": mode,
        "v_cols_used": v_cols,
        "distributed_users": distributed_users,
    }
    return df_out, meta
