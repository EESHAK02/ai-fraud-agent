# src/red_team/run_attack.py
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

from src.data_loader import PROC_DIR
from src.adversary.generator import generate_burst
from src.adversary.injector import inject_attacks
from src.feature_utils import add_basic_features_to_df
from src.baselines.iso_forest import score_and_eval  # reuses trained model

OUT_DIR = Path("experiments/red_team_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Which top-K to measure (simulates analyst capacity)
K = 200

def main():
    # 1) load test split (already featured) to get timeline + V cols
    base = pd.read_parquet(PROC_DIR / "test.parquet")
    base = base.sort_values("timestamp").reset_index(drop=True)

    # 2) choose an attack start time slightly after test begins
    start_time = base["timestamp"].min() + pd.Timedelta(minutes=5)

    # 3) generate burst
    att = generate_burst(
        start_time=start_time.to_pydatetime(),
        burst_length=300, interval_seconds=2, amount=49.99,
        merchant_pool_size=200, reuse_device=True
    )
    att["timestamp"] = pd.to_datetime(att["timestamp"])

    # 4) inject into base
    combined = inject_attacks(base, att)

    # 5) recompute features on combined (for engineered columns)
    combined_feat = add_basic_features_to_df(combined)

    # carry through V1..V28 if present in base; injected rows won't have V's -> fill 0
    vcols = [c for c in base.columns if c.startswith("V")]

    if vcols:
        # 1) merge V-cols from base by txn_id, only if any are missing
        if not set(vcols).issubset(combined_feat.columns):
            combined_feat = combined_feat.merge(
                base[["txn_id"] + vcols], on="txn_id", how="left"
            )

        # 2) ensure all expected vcols exist (create if missing), then fill NaN
        missing = [c for c in vcols if c not in combined_feat.columns]
        for c in missing:
            combined_feat[c] = 0.0

        existing = [c for c in vcols if c in combined_feat.columns]
        if existing:
            combined_feat[existing] = combined_feat[existing].fillna(0.0)

    # 6) save combined features
    attacked_path = PROC_DIR / "test_with_attack_features.parquet"
    combined_feat.to_parquet(attacked_path, index=False)

    # 7) score and basic eval (prints PR-AUC/ROC-AUC/Precision@K)
    scored = score_and_eval(attacked_path, out_path=PROC_DIR / "test_with_attack_scored.parquet", K=K)

    # 8) extra red-team metrics
    # detection latency: time from attack start to first flagged (in top-K window) — we’ll compute top-K set
    scored = scored.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored)+1)

    # precision@K already printed; now recall on injected rows
    topk = scored.head(K)
    is_attack = (scored["label"] == 1) & (scored["user_id"].str.startswith("adv_user"))
    is_attack_topk = (topk["label"] == 1) & (topk["user_id"].str.startswith("adv_user"))

    total_attack = is_attack.sum()
    caught_attack_topk = is_attack_topk.sum()
    recall_attacks_in_topk = (caught_attack_topk / total_attack) if total_attack > 0 else 0.0

    # detection latency: first time any injected row appears in the ranking (smaller rank = sooner)
    first_attack_rank = scored[is_attack]["rank"].min() if total_attack > 0 else np.nan
    first_attack_time = scored[is_attack]["timestamp"].min() if total_attack > 0 else pd.NaT
    latency_seconds = (first_attack_time - start_time).total_seconds() if total_attack > 0 else np.nan

    print(f"[RED TEAM] total injected txns: {total_attack}")
    print(f"[RED TEAM] attacks in top-{K}: {caught_attack_topk} (recall@{K}={recall_attacks_in_topk:.3f})")
    print(f"[RED TEAM] first attack rank: {first_attack_rank}")
    print(f"[RED TEAM] detection latency (s): {latency_seconds:.1f}")

    # 9) save a compact report CSV
    rep = pd.DataFrame([{
        "burst_length": 300,
        "interval_s": 2,
        "merchant_pool_size": 200,
        "amount": 49.99,
        "K": K,
        "total_attack": int(total_attack),
        "caught_attack_topK": int(caught_attack_topk),
        "recall_attacks_in_topK": float(recall_attacks_in_topk),
        "first_attack_rank": float(first_attack_rank if pd.notna(first_attack_rank) else -1),
        "detection_latency_s": float(latency_seconds if latency_seconds == latency_seconds else -1)
    }])
    rep_path = OUT_DIR / "attack_report.csv"
    rep.to_csv(rep_path, index=False)
    print("Saved red-team report →", rep_path)

if __name__ == "__main__":
    main()
