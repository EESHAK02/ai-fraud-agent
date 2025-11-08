# src/explain/report.py
from pathlib import Path
from typing import Callable, Dict, List
import pandas as pd
import numpy as np
import json
from datetime import datetime
from src.adversary.detectors_explain import explain_rule_burst_row
from src.explain.agent import explain_row  # uses LLM if provided, else fallback
from src.adversary.detectors import rule_burst_score  

OUT_DIR = Path("experiments/red_team_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _facts_for_llm(row: Dict) -> Dict:
    keep = [
        "txn_id","timestamp","user_id","merchant_id","amount","label",
        "combined_score","anomaly_score","rule_score",
        "user_txns_last_window","same_amt_last_window","device_user_count_24h",
        "z_amount_user","time_since_last_txn_user","hour","dayofweek"
    ]
    return {k: row.get(k) for k in keep if k in row}

def _ensure_combined(df: pd.DataFrame, alpha: float = 0.6) -> pd.DataFrame:
    import numpy as np
    out = df.copy()
    # rule_score
    if "rule_score" not in out.columns:
        out["rule_score"] = rule_burst_score(out)
    # anomaly_score may be missing if someone didn’t run IF; fall back to zeros
    if "anomaly_score" not in out.columns:
        out["anomaly_score"] = 0.0

    def norm(x):
        x = np.asarray(x, dtype=float)
        mx, mn = np.nanmax(x), np.nanmin(x)
        return np.zeros_like(x) if mx == mn else (x - mn) / (mx - mn)

    out["combined_score"] = alpha * norm(out["anomaly_score"].values) + (1 - alpha) * norm(out["rule_score"].values)
    return out

def build_incident_report(
    scored: pd.DataFrame,
    start_time: pd.Timestamp | None,
    provider_llm,
    top_n: int = 25,
    K: int = 200,
    file_name: str = "incident_report.md",
) -> Path:
    # NEW: ensure combined_score exists
    df = _ensure_combined(scored, alpha=0.6).sort_values("combined_score", ascending=False).reset_index(drop=True)
    top = df.head(top_n)
    # KPIs (if present from your run)
    is_attack = (df.get("label", 0) == 1) & df["user_id"].astype(str).str.startswith("adv_user")
    total_attack = int(is_attack.sum())
    topk = df.head(K)
    caught_topk = int(((topk.get("label",0)==1) & topk["user_id"].astype(str).str.startswith("adv_user")).sum())
    recall_at_k = (caught_topk / total_attack) if total_attack > 0 else 0.0
    if start_time is not None:
        attacks_in_topk = topk[topk["user_id"].astype(str).str.startswith("adv_user")]
        if not attacks_in_topk.empty:
            first_detected_time = pd.to_datetime(attacks_in_topk["timestamp"].min())
            latency_s = float((first_detected_time - pd.to_datetime(start_time)).total_seconds())
            first_rank = int(attacks_in_topk["rank"].min()) if "rank" in attacks_in_topk.columns else int(attacks_in_topk.index.min()+1)
        else:
            latency_s, first_rank = None, None
    else:
        latency_s, first_rank = None, None

    # Aggregate quick stats for LLM summary
    burst_rows = df[df["user_id"].astype(str).str.startswith("adv_user")]
    agg = {
        "total_rows": len(df),
        "total_injected": int(len(burst_rows)),
        "amount_mode": float(burst_rows["amount"].mode().iloc[0]) if not burst_rows.empty else None,
        "median_interval_s": None,  # timeline not explicit per-row; omitted
        "distinct_merchants": int(burst_rows["merchant_id"].nunique()) if "merchant_id" in burst_rows else 0,
        "recall_at_K": float(recall_at_k),
        "first_attack_rank": int(first_rank) if first_rank is not None else -1,
        "latency_seconds": float(latency_s) if latency_s is not None else -1,
        "K": int(K),
    }

    # LLM overall narrative
    overall_prompt = (
        "You are a senior fraud analyst. Given the summary JSON and the top flagged rows, "
        "write a concise incident summary (<=120 words) and a checklist of next actions. "
        "Return JSON with keys: overall_summary, next_actions.\n\n"
        f"Summary: {json.dumps(agg)}\n"
        "TopRows: " + json.dumps([_facts_for_llm(r) for r in top.to_dict(orient="records")], default=str)
    )
    if provider_llm is not None:
        try:
            resp = provider_llm(overall_prompt)
            try:
                overall = json.loads(resp)
            except Exception:
                overall = {"overall_summary": resp, "next_actions": "- Investigate manually"}
        except Exception as e:
            overall = {"overall_summary": f"LLM error: {e}", "next_actions": "- Manual review"}
    else:
        overall = {
            "overall_summary": (
                f"Top-{K} review caught {caught_topk}/{total_attack} injected txns "
                f"(recall@K={recall_at_k:.3f}). "
                + (f"First detected at rank {first_rank}, latency {latency_s:.1f}s." if first_rank else "No injected txn in Top-K.")
            ),
            "next_actions": "- Lower IF weight / raise rule weight\n- Expand K for analyst queue\n- Add merchant-fanout & velocity thresholds"
        }

    # Per-row explanations
    lines = []
    for _, r in top.iterrows():
        row = r.to_dict()
        rule_reason = explain_rule_burst_row(r)
        exp = explain_row(row, llm=None) if provider_llm is None else explain_row(row, llm=provider_llm)
        lines.append(
            f"- **rank {int(r.get('rank', _+1))}** | txn `{row.get('txn_id')}` | "
            f"user `{row.get('user_id')}` → ${row.get('amount'):.2f} | "
            f"combined={row.get('combined_score'):.3f} | label={row.get('label',0)}\n"
            f"  - rule: {rule_reason}\n"
            f"  - agent: {exp.get('plain_explanation')}\n"
        )

    # Write markdown
    p = OUT_DIR / file_name
    with p.open("w", encoding="utf-8") as f:
        f.write(f"# Incident Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Summary\n")
        f.write(overall.get("overall_summary","") + "\n\n")
        f.write("### KPIs\n")
        f.write(f"- Injected: **{total_attack}**\n")
        f.write(f"- In Top-{K}: **{caught_topk}** (recall@K={recall_at_k:.3f})\n")
        f.write(f"- First attack rank: **{first_rank if first_rank is not None else '—'}**\n")
        f.write(f"- Detection latency (s): **{latency_s if latency_s is not None else '—'}**\n\n")
        f.write("## Top flagged transactions\n")
        f.write("\n".join(lines) + "\n\n")
        f.write("## Next actions\n")
        f.write(overall.get("next_actions","") + "\n")
    return p
