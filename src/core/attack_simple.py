# src/core/attack_simple.py
import pandas as pd
import numpy as np

def inject_simple_burst(df, n=150, amount=9999.0, adv_user="adv_user_test",
                        merchants_pool=10, interval_s=1, t0_offset_minutes=10,
                        clone_template=True):
    """
    Inject a simple, obvious attack: n transactions for a single adv_user
    with a large Amount (default 9999) spaced by interval_s seconds.
    If V* (PCA) columns exist, this clones a template row so V1..V28 remain.
    Returns (df_out, meta)
    """
    df = df.copy()

    # determine base timestamp to start the attack
    if "timestamp" in df.columns:
        start = pd.to_datetime(df["timestamp"].min()) + pd.Timedelta(minutes=t0_offset_minutes)
    elif "Time" in df.columns:
        start = pd.to_datetime("2013-01-01") + pd.Timedelta(df["Time"].min(), unit="s") + pd.Timedelta(minutes=t0_offset_minutes)
    else:
        start = pd.Timestamp("2013-01-01 00:00:00")

    # try to find a row that has V* columns to clone (to keep PCA fields realistic)
    template = None
    if clone_template:
        for c in df.columns:
            if str(c).lower().startswith("v"):
                template = df.sample(1, random_state=42).iloc[0]
                break

    merchants = [f"m_simple_{i}" for i in range(merchants_pool)]
    rows = []
    for i in range(n):
        ts = start + pd.Timedelta(seconds=i * interval_s)
        base = {}
        if template is not None:
            # copy PCA + any other fields
            base = template.to_dict()
        base.update({
            "txn_id": f"adv_simple_{i}_{np.random.randint(1_000_000)}",
            "timestamp": ts,
            "user_id": adv_user,
            "merchant_id": merchants[i % merchants_pool],
            "Amount": float(amount),
            "Class": 1,        # mark as fraud label if you want
            "is_attack": 1,    # explicit attack flag
        })
        rows.append(base)

    inj = pd.DataFrame(rows)
    out = pd.concat([df, inj], ignore_index=True, sort=False)
    meta = {"t0": start, "n_injected": n}
    return out.reset_index(drop=True), meta
