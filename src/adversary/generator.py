# src/adversary/generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid

def generate_burst(
    start_time: datetime,
    burst_length: int = 300,
    interval_seconds: int = 2,
    amount: float = 49.99,
    merchant_pool_size: int = 200,
    user_prefix: str = "adv_user",
    reuse_device: bool = True,
    base_country: str = "US",
):
    merchants = [f"m_{i}" for i in range(merchant_pool_size)]
    device_id = f"dev_{uuid.uuid4().hex[:8]}" if reuse_device else None
    rows = []
    t = start_time
    for i in range(burst_length):
        rows.append({
            "txn_id": str(uuid.uuid4()),
            "user_id": f"{user_prefix}_{i % 5}",
            "merchant_id": np.random.choice(merchants),
            "amount": amount,
            "timestamp": t,
            "country": base_country,
            "currency": "USD",
            "channel": "card",
            "device_id": device_id if reuse_device else f"dev_{uuid.uuid4().hex[:8]}",
            "lat": None, "lon": None, "mcc": "5311",
            "label": 1  # mark injected as fraud for eval
        })
        t = t + timedelta(seconds=interval_seconds)
    return pd.DataFrame(rows)
