# src/adversary/save_burst.py
import pandas as pd
from pathlib import Path
from src.adversary.generator import generate_burst

SYN_DIR = Path("data/synthetic")
SYN_DIR.mkdir(parents=True, exist_ok=True)

def save_burst_profile(start_time, burst_length=300, interval_seconds=2, amount=49.99, merchant_pool_size=200, filename="burst_profile.csv"):
    df = generate_burst(start_time=start_time, burst_length=burst_length, interval_seconds=interval_seconds,
                        amount=amount, merchant_pool_size=merchant_pool_size)
    out = SYN_DIR / filename
    df.to_csv(out, index=False)
    print("Saved burst:", out)
    return out
