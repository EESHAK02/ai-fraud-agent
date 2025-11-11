# src/explain/agent.py
import json, requests, textwrap

OLLAMA_URL = "http://localhost:11434/api/generate"  # default Ollama endpoint

DEFAULT_MODEL = "phi3:mini"  # pull once via: ollama pull phi3:mini
MAX_TOKENS = 300  # keep it short & snappy

PROMPT = """You are a concise fraud analyst.
Explain in <=70 words why THIS transaction was flagged, using ONLY the facts given.
Then give three single-line triage steps (bullets). Do not return JSON; plain text only.

Facts:
{facts}
"""

def _format_facts(row: dict) -> str:
    keep = [
        "txn_id","timestamp","user_id","merchant_id","Amount","Class",
        "vel_5m","same_amt_5m","fanout_5m","z_amount",
        "anomaly_score","rule_score","combined_score"
    ]
    parts = []
    for k in keep:
        if k in row:
            parts.append(f"{k}: {row[k]}")
    return "\n".join(parts)

def llm_explain_row(row: dict, model: str = DEFAULT_MODEL, timeout: float = 30.0) -> str:
    """Return plain-text explanation via Ollama; raise-friendly but safe to call."""
    facts = _format_facts(row)
    prompt = PROMPT.format(facts=facts)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": MAX_TOKENS
        }
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("response", "").strip()
    # light cleanup
    text = textwrap.shorten(text.replace("\r", " ").replace("\n\n", "\n"), width=900, placeholder="…")
    return text

def is_ollama_up(timeout: float = 3.0) -> bool:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=timeout)
        return True
    except Exception:
        return False

