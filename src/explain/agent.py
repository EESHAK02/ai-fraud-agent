# # src/explain/agent.py
# from typing import Callable, Dict, Any
# import json
# import os

# # ---------- deterministic fallback (no LLM needed) ----------
# def _facts(row: Dict[str, Any]) -> Dict[str, Any]:
#     keys = [
#         "txn_id","user_id","merchant_id","amount","timestamp",
#         "anomaly_score","rule_score","combined_score",
#         "z_amount_user","time_since_last_txn_user",
#         "user_txns_last_window","same_amt_last_window","device_user_count_24h",
#         "hour","dayofweek"
#     ]
#     return {k: row.get(k) for k in keys if k in row}

# def _deterministic(row: Dict[str, Any]) -> Dict[str, str]:
#     f = _facts(row)
#     ut = int(f.get("user_txns_last_window") or 0)
#     sa = int(f.get("same_amt_last_window") or 0)
#     du = int(f.get("device_user_count_24h") or 0)
#     reasons = []
#     if ut > 5: reasons.append(f"{ut} txns by user in 5 min")
#     elif ut > 2: reasons.append(f"{ut} txns in 5 min (high velocity)")
#     if sa > 5: reasons.append(f"amount ${float(f.get('amount',0)):.2f} repeated {sa}×")
#     elif sa > 2: reasons.append("repeated amount across merchants")
#     if du > 3: reasons.append(f"device used by {du} users in 24h")
#     if not reasons:
#         a = f.get("anomaly_score"); r = f.get("rule_score")
#         if a and a > 0.75: reasons.append(f"high anomaly score ({a:.2f})")
#         elif r and r > 0.5: reasons.append("rule score indicates burstiness")
#         else: reasons.append("unusual vs baseline behavior")
#     triage = "- Verify device ownership\n- Check merchant risk flags\n- Rate limit further identical txns"
#     return {"plain_explanation": " | ".join(reasons), "triage_suggestions": triage}

# # ---------- LLM call helpers (import inside to avoid import-time errors) ----------
# def make_openai(model: str = "gpt-4o-mini") -> Callable[[str], str]:
#     from openai import OpenAI  # imported here so module import never fails
#     api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
#     if not api_key:
#         raise RuntimeError("OPENAI_API_KEY not set")
#     client = OpenAI(api_key=api_key)
#     def _call(prompt: str) -> str:
#         resp = client.chat.completions.create(
#             model=model,
#             messages=[{"role":"user","content":prompt}],
#             max_tokens=300,
#             temperature=0.2,
#         )
#         return resp.choices[0].message.content
#     return _call

# def make_ollama(host: str = "http://localhost:11434", model: str = "llama3") -> Callable[[str], str]:
#     import requests
#     url = host.rstrip("/") + "/api/generate"
#     def _call(prompt: str) -> str:
#         r = requests.post(url, json={"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": 300}}, timeout=60)
#         r.raise_for_status()
#         return r.json().get("response", "")
#     return _call

# # ---------- main entry: explain_row ----------
# _PROMPT = (
#     "You are a concise fraud analyst. Using the JSON facts, explain (<=80 words) "
#     "why this txn was flagged and give 3 short triage actions. Return JSON with "
#     "keys plain_explanation and triage_suggestions.\nFacts:\n{facts}"
# )

# def explain_row(row: Dict[str, Any], llm: Callable[[str], str] | None = None) -> Dict[str, str]:
#     """
#     Always available. If llm is None, returns deterministic explanation.
#     If llm is provided (callable), attempts LLM explanation; on any error falls back.
#     """
#     if llm is None:
#         return _deterministic(row)
#     facts = _facts(row)
#     prompt = _PROMPT.format(facts=json.dumps(facts, default=str))
#     try:
#         txt = llm(prompt)
#         try:
#             out = json.loads(txt)
#             if isinstance(out, dict) and "plain_explanation" in out:
#                 return out
#         except Exception:
#             pass
#         return {"plain_explanation": txt, "triage_suggestions": "- Investigate manually"}
#     except Exception as e:
#         fb = _deterministic(row)
#         fb["plain_explanation"] = f"LLM error: {e}. " + fb["plain_explanation"]
#         return fb
    
# # ---- Ollama utilities: list models & factory by name ----
# def list_ollama_models(host: str = "http://localhost:11434"):
#     import requests
#     url = host.rstrip("/") + "/api/tags"
#     try:
#         r = requests.get(url, timeout=10)
#         r.raise_for_status()
#         data = r.json()
#         return [m.get("name") for m in data.get("models", []) if m.get("name")]
#     except Exception:
#         return []

# def make_ollama_from_name(name: str, host: str = "http://localhost:11434"):
#     # convenience wrapper that reuses our non-streaming call
#     return make_ollama(host=host, model=name)

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

