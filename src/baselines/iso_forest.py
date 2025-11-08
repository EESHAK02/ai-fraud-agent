# src/baselines/iso_forest.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data_loader import PROC_DIR

# core engineered features
BASE_FEATS = ["amount","z_amount_user","time_since_last_txn_user","txns_last_1h_user","hour","dayofweek"]

def _feature_list(df: pd.DataFrame):
    vcols = [c for c in df.columns if c.startswith("V")]
    return BASE_FEATS + vcols

def train_if(model_path="models/if_model.joblib"):
    train = pd.read_parquet(PROC_DIR / "train.parquet")
    feats = _feature_list(train)
    X = train[feats].astype(float).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=300, max_samples="auto", contamination="auto",
        random_state=42, n_jobs=-1
    )
    model.fit(Xs)

    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "features": feats}, model_path)
    print("Saved IF model →", model_path)

def score_and_eval(df_path: Path, model_path="models/if_model.joblib",
                   out_path: Path | None = None, K: int = 200):
    obj = joblib.load(model_path)
    model, scaler, feats = obj["model"], obj["scaler"], obj["features"]

    df = pd.read_parquet(df_path)
    X = df[feats].astype(float).values
    Xs = scaler.transform(X)
    # higher = more anomalous
    scores = -model.score_samples(Xs)
    df["anomaly_score"] = scores

    if "label" in df.columns:
        # quick eval
        y = df["label"].astype(int).values
        try:
            pr_auc = average_precision_score(y, scores)
        except Exception:
            pr_auc = float("nan")
        try:
            roc_auc = roc_auc_score(y, scores)
        except Exception:
            roc_auc = float("nan")
        # Precision@K
        order = np.argsort(scores)[::-1]
        topk = y[order][:K]
        prec_at_k = topk.mean() if len(topk) > 0 else float("nan")
        print(f"PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | Precision@{K}: {prec_at_k:.4f}")

    if out_path is not None:
        df.to_parquet(out_path, index=False)
        print("Saved scored file →", out_path)

    return df

if __name__ == "__main__":
    train_if()
    score_and_eval(PROC_DIR/"val.parquet", out_path=PROC_DIR/"val_scored.parquet")
    score_and_eval(PROC_DIR/"test.parquet", out_path=PROC_DIR/"test_scored.parquet")
