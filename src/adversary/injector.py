# src/adversary/injector.py
import pandas as pd

def inject_attacks(real_df: pd.DataFrame, attack_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([real_df, attack_df], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
