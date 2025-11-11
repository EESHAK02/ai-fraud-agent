# src/explain/report_simple.py
from __future__ import annotations
import json
import math
from typing import Dict, List, Optional
import pandas as pd

# --- Lightweight Ollama HTTP helpers (no external deps) -----------------------
import urllib.request
import urllib.error

def _http_post(url: str, payload: dict, timeout: float = 8.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _http_get(url: str, timeout: float = 2.0) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _ollama_up() -> bool:
    try:
        _http_get("http://localhost:11434/api/tags", timeout=1.0)
        return True
    except Exception:
        return False

def _ollama_generate(model: str, prompt: str) -> str:
    try:
        resp = _http_post(
            "http://localhost:11434/api/generate",
            {"model": model, "prompt": prompt, "stream": False},
            timeout=30.0,
        )
        return resp.get("response", "").strip()
    except Exception:
        return ""


# --- Public API ----------------------------------------------------------------

def build_incident_summary(scored: pd.DataFrame, k: int = 200, use_llm: bool = False) -> str:
    """
    Compact markdown report (used for download). Pure pandas, no LLM needed.
    Expects columns: txn_id, user_id, merchant_id, Amount, combined_score, Class (optional),
                     is_attack (optional)
    """
    if scored is None or len(scored) == 0:
        return "# Incident Report\n\n_No data._"

    top = scored.nlargest(k, "combined_score").copy()
    n_all = len(scored)
    thr = float(top["combined_score"].iloc[-1]) if len(top) else float("nan")

    # optional attack metrics
    injected = int(scored.get("is_attack", pd.Series(False, index=scored.index)).sum())
    hits = int(top.get("is_attack", pd.Series(False, index=top.index)).sum())
    recall_k = (hits / injected) if injected > 0 else 0.0

    lines = []
    lines.append("# Incident Report\n")
    lines.append("**Scope**")
    lines.append(f"- Total transactions scored: **{n_all:,}**")
    lines.append(f"- Top-{k} threshold (combined_score): **{thr:.3f}**")
    if injected > 0:
        lines.append(f"- Injected attacks: **{injected:,}** | In Top-{k}: **{hits:,}** | recall@{k}=**{recall_k:.3f}**")
    lines.append("")

    # top alerts list (sample)
    lines.append("**Top alerts (sample)**\n")
    cols = ["txn_id", "user_id", "merchant_id", "Amount", "combined_score", "Class"]
    show = [c for c in cols if c in top.columns]
    for _, r in top[show].head(30).iterrows():
        amt = f"${r['Amount']:.2f}" if "Amount" in r else ""
        score = f"{r['combined_score']:.3f}" if "combined_score" in r else ""
        uid = r.get("user_id", "")
        mid = r.get("merchant_id", "")
        tx  = r.get("txn_id", "")
        lines.append(f"- {tx} | user {uid} | {amt} | score={score} | {mid}")

    return "\n".join(lines)


def build_llm_overview(top_df: pd.DataFrame, model: Optional[str] = "phi3:mini") -> dict:
    """
    Return {'narrative': str, 'triage': [str,...]}.
    Uses Ollama if available + model provided; otherwise falls back to a clear default.
    """
    triage_default = [
        "Contact customer for step-up verification on highest scores.",
        "Rate-limit high-velocity accounts for the next 24 hours.",
        "Review merchants with concentrated alerts and recent new-merchant activity.",
        "Lower per-transaction limits for flagged users until verified.",
        "Add watchlist rules for repetitive small-amount bursts ($1–$50).",
    ]

    if top_df is None or top_df.empty:
        return {
            "narrative": "No high-scoring alerts available for summary.",
            "triage": []
        }

    # If no LLM wanted/available, return defaults
    if not model or not _ollama_up():
        return {
            "narrative": (
                "Multiple high-scoring anomalies are concentrated among a small set of users and merchants. "
                "Drivers include unusual amounts relative to user baselines, short-interval velocity, and low-probability hours. "
                "Prioritize clustered entities and accounts with repeated Top-K appearances."
            ),
            "triage": triage_default,
        }

    # Build compact facts for LLM
    cols = ["txn_id","user_id","merchant_id","Amount","hour","vel_5m","fanout_5m","combined_score"]
    cols = [c for c in cols if c in top_df.columns]
    sample = top_df[cols].head(100)
    facts = sample.to_string(index=False)

    prompt = f"""
    You are a senior fraud analyst. Given these top alerts (columns: {", ".join(cols)}):
    {facts}

    1) Write a concise 4–6 sentence incident overview for leadership (no tables).
    2) Then list 5 short, actionable triage steps as bullet points.

    Return plain text; start with the overview, then a line '---', then the five bullets (one per line starting with '- ').
    """
    
    text = _ollama_generate(model, prompt).strip()
    if not text:
        return {
            "narrative": (
                "Multiple high-scoring anomalies with signs of bursty behavior and off-hour usage. "
                "Focus on repeated offenders and merchants with many hits."
            ),
            "triage": triage_default,
        }

    if "---" in text:
        overview, bullets = text.split("---", 1)
        triage = [b.strip(" -•\n\r\t") for b in bullets.strip().splitlines() if b.strip()]
    else:
        overview = text
        triage = triage_default

    return {"narrative": overview.strip(), "triage": triage[:5] or triage_default}
