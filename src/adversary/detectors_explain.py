# src/adversary/detectors_explain.py
import pandas as pd

def explain_rule_burst_row(row: pd.Series) -> str:
    """
    Turn rule features into a concise, human-readable reason.
    Expects these columns if available:
      user_txns_last_window, same_amt_last_window, device_user_count_24h, amount, user_id, device_id
    """
    bits = []
    ut = int(row.get("user_txns_last_window", 0) or 0)
    sa = int(row.get("same_amt_last_window", 0) or 0)
    du = int(row.get("device_user_count_24h", 0) or 0)
    amt = row.get("amount", None)
    if ut > 3:
        bits.append(f"{ut} txns by user in 5m")
    if sa > 3 and amt is not None:
        bits.append(f"amount ${float(amt):.2f} repeated {sa}×")
    if du > 5:
        bits.append(f"device used by {du} users/24h")
    return " | ".join(bits) if bits else "no strong burst pattern"
