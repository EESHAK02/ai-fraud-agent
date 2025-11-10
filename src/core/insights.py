# src/core/insights.py
import pandas as pd
import numpy as np

def build_user_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-user baselines for comparison in Inspect.
    Requires columns: user_id, merchant_id, Amount, hour, vel_5m, fanout_5m, txn_id
    """
    d = df.copy()

    # --- per-user amount stats
    g = d.groupby("user_id", observed=True)
    prof = g["Amount"].agg(mu_amt="mean", sd_amt="std", p95_amt=lambda x: x.quantile(0.95))
    prof["sd_amt"] = prof["sd_amt"].replace(0, np.nan)

    # --- per-user velocity / fanout typical levels
    prof2 = g[["vel_5m", "fanout_5m"]].agg(
        med_vel=("vel_5m", "median"),
        p95_vel=("vel_5m", lambda x: x.quantile(0.95)),
        med_fan=("fanout_5m", "median"),
        p95_fan=("fanout_5m", lambda x: x.quantile(0.95)),
    )
    prof = prof.join(prof2, how="left")

    # --- hourly distribution P(hour|user)
    hour_counts = d.groupby(["user_id","hour"], observed=True)["txn_id"].count().rename("hour_ct")
    hour_total  = hour_counts.groupby("user_id").sum()
    hour_prob   = (hour_counts / hour_total).reset_index()
    hour_pivot  = (
        hour_prob.pivot(index="user_id", columns="hour", values="hour_ct")
        .fillna(0.0)
    )
    hour_pivot = hour_pivot.div(hour_pivot.sum(axis=1).replace(0, 1), axis=0)
    hour_pivot.columns = [f"p_hour_{int(h)}" for h in hour_pivot.columns]
    prof = prof.join(hour_pivot, how="left")

    # --- top merchants per user (store as Python set)
    top_merch = (
        d.groupby(["user_id","merchant_id"], observed=True)["txn_id"].count()
         .reset_index()
         .sort_values(["user_id","txn_id"], ascending=[True, False])
    )
    k = 10
    top_k = top_merch.groupby("user_id").head(k)

    # Initialize column with empty sets (object dtype), then fill only existing users
    prof["top_merchants"] = pd.Series([set()] * len(prof), index=prof.index, dtype="object")
    # Build a series mapping user_id -> set(...)
    top_sets = top_k.groupby("user_id")["merchant_id"].apply(lambda s: set(s.tolist()))
    # Assign without fillna
    prof.loc[top_sets.index, "top_merchants"] = top_sets.astype("object")

    return prof


def compare_txn_to_user(row: pd.Series, profiles: pd.DataFrame) -> dict:
    uid = row["user_id"]
    pr = profiles.loc[uid] if uid in profiles.index else None

    out = {
        "amount": float(row["Amount"]),
        "hour": int(row["hour"]),
        "vel_5m": int(row["vel_5m"]),
        "fanout_5m": int(row["fanout_5m"]),
        "merchant_id": str(row["merchant_id"]),
    }

    if pr is None:
        out.update({
            "z_amount": None,
            "amount_vs_mu_sd": "n/a",
            "vel_vs_typical": "n/a",
            "fanout_vs_typical": "n/a",
            "hour_prob": None,
            "merchant_familiar": None
        })
        return out

    mu = pr.get("mu_amt", np.nan)
    sd = pr.get("sd_amt", np.nan)
    z = (row["Amount"] - mu) / (sd if pd.notnull(sd) and sd != 0 else np.nan)
    out["z_amount"] = None if pd.isnull(z) else float(z)

    out["amount_vs_mu_sd"] = (
        f"{mu:.2f} ± {sd:.2f}" if pd.notnull(mu) and pd.notnull(sd) else "n/a"
    )

    med_vel = pr.get("med_vel", np.nan); p95_vel = pr.get("p95_vel", np.nan)
    out["vel_vs_typical"] = (
        f"{row['vel_5m']} vs med {med_vel:.0f} (p95 {p95_vel:.0f})"
        if pd.notnull(med_vel) else "n/a"
    )

    med_fan = pr.get("med_fan", np.nan); p95_fan = pr.get("p95_fan", np.nan)
    out["fanout_vs_typical"] = (
        f"{row['fanout_5m']} vs med {med_fan:.0f} (p95 {p95_fan:.0f})"
        if pd.notnull(med_fan) else "n/a"
    )

    hp_col = f"p_hour_{row['hour']}"
    ph = pr.get(hp_col, np.nan)
    out["hour_prob"] = None if pd.isnull(ph) else float(ph)

    top_merch = pr["top_merchants"] if "top_merchants" in pr else set()
    out["merchant_familiar"] = bool(row["merchant_id"] in top_merch)

    return out
