# src/split.py
import pandas as pd
from pathlib import Path

from data_loader import PROC_DIR

def time_split(in_path: Path = None, train_frac=0.70, val_frac=0.15):
    if in_path is None:
        in_path = PROC_DIR / "creditcard_features.parquet"
    df = pd.read_parquet(in_path).sort_values("timestamp").reset_index(drop=True)
    n = len(df); i1 = int(n*train_frac); i2 = int(n*(train_frac+val_frac))
    df.iloc[:i1].to_parquet(PROC_DIR/"train.parquet", index=False)
    df.iloc[i1:i2].to_parquet(PROC_DIR/"val.parquet", index=False)
    df.iloc[i2:].to_parquet(PROC_DIR/"test.parquet", index=False)
    print("Saved splits to", PROC_DIR)

if __name__ == "__main__":
    time_split()
