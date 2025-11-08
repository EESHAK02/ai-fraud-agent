# app/streamlit_app.py
"""
Streamlit app (clean, tabbed layout) for Agentic Fraud Detector — Red Team Playground.

Replace previous app with this file. Run:
    streamlit run app/streamlit_app.py
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import importlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import PROC_DIR
from src.feature_utils import add_basic_features_to_df
from src.adversary.generator import generate_burst
from src.adversary.injector import inject_attacks
from src.adversary.detectors import rule_burst_score
from src.adversary.detectors_explain import explain_rule_burst_row
from src.baselines.iso_forest import score_and_eval
import src.explain.agent as agent  # explanation agent (fallback rule-only if no LLM)
agent = importlib.reload(agent)
from src.explain.report import build_incident_report

st.set_page_config(page_title="Agentic Fraud Detector", layout="wide")
st.title("Agentic Fraud Detector — Red Team Playground")

# helper loader
def safe_load(p: Path):
    try:
        return pd.read_parquet(p)
    except Exception:
        return None

# Load files (if present)
val_scored = safe_load(PROC_DIR / "val_scored.parquet")
test_scored = safe_load(PROC_DIR / "test_scored.parquet")
attacked_scored = safe_load(PROC_DIR / "test_with_attack_scored.parquet")
test_base = safe_load(PROC_DIR / "test.parquet")

# Tabs
tab_overview, tab_redteam, tab_inspect, tab_reports = st.tabs(["Overview", "Red Team", "Inspector", "Reports"])

# -------------------------
# Overview Tab: quick health
# -------------------------
with tab_overview:
    st.header("Overview — quick health & top anomalies")
    c1, c2, c3 = st.columns(3)
    def show_kpis(df, title):
        if df is None:
            return (title, "n/a", "n/a")
        n = len(df)
        n_fraud = int(df['label'].sum()) if 'label' in df.columns else 0
        mean_score = float(df['anomaly_score'].mean()) if 'anomaly_score' in df.columns else np.nan
        return (title, f"{n:,}", f"{n_fraud} frauds — mean score {mean_score:.3f}")

    k1 = show_kpis(val_scored, "Validation (scored)")
    k2 = show_kpis(test_scored, "Test (scored)")
    k3 = show_kpis(attacked_scored, "Test (attacked, scored)")

    c1.metric(k1[0], k1[1], k1[2])
    c2.metric(k2[0], k2[1], k2[2])
    c3.metric(k3[0], k3[1], k3[2])

    st.markdown("**Top anomalies (recent scored test)** — highest anomaly_score")
    df_table = attacked_scored if attacked_scored is not None else (test_scored if test_scored is not None else val_scored)
    if df_table is None:
        st.info("No scored dataset found. Use the Red Team tab or run scorer to create one.")
    else:
        # show a concise table
        top = df_table.sort_values("anomaly_score", ascending=False).head(50)
        st.dataframe(top[["txn_id","timestamp","user_id","merchant_id","amount","anomaly_score","label"]], height=300)

    st.markdown("---")
    st.write("**How to interpret:**\n"
             "- `Top anomalies`: what the ML detector (IF) calls unusual.\n"
             "- Use Red Team to inject bursts and see whether they appear here.\n"
             "- If injected txns do not appear here, try lowering IF weight (alpha) in Red Team tab.")

# -------------------------
# Red Team Tab
# -------------------------
with tab_redteam:
    st.header("Red Team — generate attacks & evaluate")
    st.markdown("Configure a synthetic attack burst and run it end-to-end (generate → inject → feature rebuild → score → evaluate).")

    # controls
    colA, colB = st.columns(2)
    with colA:
        burst_length = st.number_input("Burst length (txns)", min_value=5, max_value=5000, value=300, step=5)
        interval_seconds = st.number_input("Interval between txns (s)", min_value=1, max_value=3600, value=2)
        merchant_pool_size = st.number_input("Merchant pool size", min_value=1, max_value=5000, value=200)
        amount = float(st.text_input("Amount per txn (USD)", value="49.99"))
        reuse_device = st.checkbox("Reuse same device for burst (device fanout)", value=True)
    with colB:
        K = st.number_input("Analyst capacity (K)", min_value=10, max_value=2000, value=200)
        alpha = st.slider("Weight: IF anomaly score (alpha)", 0.0, 1.0, 0.6)
        beta = 1.0 - alpha
        st.caption(f"Hybrid score weights → IF: {alpha:.2f}, rule: {beta:.2f}")

    run_attack = st.button("Generate & Run Attack")

    if run_attack:
        status = st.empty()
        status.info("Generating burst and injecting into test split...")
        base = test_base.copy() if test_base is not None else None
        if base is None:
            status.error("No test split found at data/processed/test.parquet. Run the canonicalize/features/split steps.")
        else:
            start_time = base["timestamp"].min() + pd.Timedelta(minutes=5)
            st.session_state["attack_start_time"] = start_time
            attack_df = generate_burst(start_time=start_time.to_pydatetime(),
                                       burst_length=int(burst_length),
                                       interval_seconds=int(interval_seconds),
                                       amount=float(amount),
                                       merchant_pool_size=int(merchant_pool_size),
                                       reuse_device=bool(reuse_device))
            attack_df["timestamp"] = pd.to_datetime(attack_df["timestamp"])
            combined = inject_attacks(base, attack_df)
            status.info("Recomputing features on combined dataset...")
            combined_feat = add_basic_features_to_df(combined)

            # robust merge for V cols (if base has them)
            vcols = [c for c in base.columns if c.startswith("V")]
            if vcols:
                if not set(vcols).issubset(combined_feat.columns):
                    combined_feat = combined_feat.merge(base[["txn_id"] + vcols], on="txn_id", how="left")
                for c in vcols:
                    if c not in combined_feat.columns:
                        combined_feat[c] = 0.0
                combined_feat[vcols] = combined_feat[vcols].fillna(0.0)

            attacked_path = PROC_DIR / "test_with_attack_features.parquet"
            combined_feat.to_parquet(attacked_path, index=False)
            status.info("Saved combined features. Scoring...")

            # score with IF
            scored = score_and_eval(attacked_path, out_path=PROC_DIR / "test_with_attack_scored.parquet", K=int(K))

            # rule score + combined score (alpha/beta)
            scored["rule_score"] = rule_burst_score(scored)
            def norm_col(s):
                s = np.array(s, dtype=float)
                if np.nanmax(s) == np.nanmin(s):
                    return np.zeros_like(s)
                return (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s))
            scored["anomaly_score_norm"] = norm_col(scored["anomaly_score"].values)
            scored["rule_score_norm"] = norm_col(scored["rule_score"].values)
            scored["combined_score"] = alpha * scored["anomaly_score_norm"] + (1 - alpha) * scored["rule_score_norm"]

            scored = scored.sort_values("combined_score", ascending=False).reset_index(drop=True)
            scored["rank"] = np.arange(1, len(scored) + 1)

            # red-team metrics
            is_attack = (scored.get("label", 0) == 1) & scored["user_id"].astype(str).str.startswith("adv_user")
            total_attack = int(is_attack.sum())
            topk = scored.head(int(K))
            caught_topk = int(((topk.get("label", 0) == 1) & topk["user_id"].astype(str).str.startswith("adv_user")).sum())
            recall_at_k = (caught_topk / total_attack) if total_attack > 0 else 0.0
            attacks_in_topk_rows = topk[topk["user_id"].astype(str).str.startswith("adv_user")]
            if not attacks_in_topk_rows.empty:
                first_detected_time = pd.to_datetime(attacks_in_topk_rows["timestamp"].min())
                latency_seconds = (first_detected_time - pd.to_datetime(start_time)).total_seconds()
                first_rank = int(attacks_in_topk_rows["rank"].min())
            else:
                latency_seconds = None
                first_rank = None

            # display concise results
            st.subheader("Red-team summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Injected txns", f"{total_attack}")
            col2.metric(f"Attacks in Top-{K}", f"{caught_topk}", delta=f"recall@K: {recall_at_k:.3f}")
            col3.metric("First detected rank", f"{first_rank if first_rank is not None else '—'}", delta=f"latency(s): {latency_seconds if latency_seconds is not None else '—'}")

            # timeline plot
            st.subheader("Transaction timeline (colored by combined_score)")
            sample_df = scored.sort_values("timestamp")
            sample_df["ts"] = pd.to_datetime(sample_df["timestamp"])
            plot_df = sample_df if len(sample_df) <= 5000 else sample_df.sample(5000, random_state=1)
            fig = px.scatter(plot_df, x="ts", y="amount", color="combined_score",
                             hover_data=["txn_id","user_id","merchant_id","amount","combined_score","label"],
                             title="Timeline (attacks highlighted by label)")
            st.plotly_chart(fig, use_container_width=True)

            # top anomalies
            st.subheader("Top anomalies (combined score)")
            st.dataframe(scored[["rank","txn_id","timestamp","user_id","merchant_id","amount","combined_score","label"]].head(200), height=300)

            # save report
            rep = pd.DataFrame([{
                "burst_length": burst_length,
                "interval_s": interval_seconds,
                "merchant_pool_size": merchant_pool_size,
                "amount": amount,
                "K": K,
                "total_attack": total_attack,
                "caught_attack_topK": caught_topk,
                "recall_attacks_in_topK": recall_at_k,
                "first_attack_rank": first_rank if first_rank is not None else -1,
                "detection_latency_s": latency_seconds if latency_seconds is not None else -1
            }])
            rep.to_csv("experiments/red_team_results/attack_report.csv", index=False)
            status.success("Red team run complete and results saved.")

# -------------------------
# Inspector Tab: explain & triage
# -------------------------
with tab_inspect:
    st.header("Inspector — explain a flagged transaction")

    # Prefer attacked+scored; fallback to plain test_scored if needed
    scored = safe_load(PROC_DIR / "test_with_attack_scored.parquet")
    if scored is None:
        scored = safe_load(PROC_DIR / "test_scored.parquet")

    if scored is None or scored.empty:
        st.info("No scored dataset found. Run Red Team or the baseline scorer first.")
    else:
        # --- Ensure combined_score exists (compute if missing) ---
        import numpy as np
        from src.adversary.detectors import rule_burst_score

        if "combined_score" not in scored.columns:
            # add rule score if missing
            if "rule_score" not in scored.columns:
                scored["rule_score"] = rule_burst_score(scored)

            # robust min-max normalize
            def norm(x):
                x = np.asarray(x, dtype=float)
                mx, mn = np.nanmax(x), np.nanmin(x)
                return np.zeros_like(x) if mx == mn else (x - mn) / (mx - mn)

            a = scored["anomaly_score"].values if "anomaly_score" in scored.columns else np.zeros(len(scored))
            r = scored["rule_score"].values
            alpha = 0.6  # same default as Red Team tab
            scored["combined_score"] = alpha * norm(a) + (1 - alpha) * norm(r)

        scored = scored.sort_values("combined_score", ascending=False).reset_index(drop=True)

        options = scored.head(500)["txn_id"].tolist()
        pick = st.selectbox("Pick a txn to explain", options=options, index=0)
        row = scored.loc[scored["txn_id"] == pick].iloc[0]

        from src.adversary.detectors_explain import explain_rule_burst_row
        st.markdown("**Rule reasoning (deterministic):**")
        st.write(explain_rule_burst_row(row))

        st.markdown("---")
        st.markdown("**LLM explainer (agentic narrative):**")
        import src.explain.agent as agent
        local_models = agent.list_ollama_models() or ["phi3:mini"]
        model_name = st.selectbox("Ollama model", local_models, index=0)

        if st.button("Explain with Agent (Ollama)"):
            try:
                llm = agent.make_ollama_from_name(model_name)
            except Exception as e:
                llm = None
                st.warning(f"Ollama init failed: {e}. Falling back to deterministic.")
            exp = agent.explain_row(row.to_dict(), llm=llm)
            st.write("**Agent explanation:**")
            st.write(exp.get("plain_explanation", "(empty)"))
            st.write("**Triage suggestions:**")
            st.code(exp.get("triage_suggestions", ""))


with tab_reports:
    st.header("Reports — generate analyst incident report")
    scored = safe_load(PROC_DIR / "test_with_attack_scored.parquet")

    # optional: if you saved start_time in session_state during red-team run
    start_time = st.session_state.get("attack_start_time", None)

    if scored is None or scored.empty:
        st.info("Run a Red Team attack first to create an attacked+scored dataset.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            top_n = st.number_input("Top-N flagged to include", min_value=10, max_value=200, value=25, step=5)
            K_for_report = st.number_input("Analyst capacity K", min_value=50, max_value=2000, value=200, step=50)
        with col2:
            local_models = agent.list_ollama_models() or ["phi3:mini"]
            model_name = st.selectbox("Ollama model for narrative", local_models, index=0)

        if st.button("Generate Report"):
            try:
                llm = agent.make_ollama_from_name(model_name)
            except Exception as e:
                llm = None
                st.warning(f"Ollama not available: {e}. Generating deterministic report.")

            path = build_incident_report(
                scored=scored,
                start_time=start_time,
                provider_llm=llm,
                top_n=int(top_n),
                K=int(K_for_report),
            )
            st.success(f"Report saved → {path}")
            st.markdown(f"[Open report]({path.as_posix()})")



st.caption("Agentic Fraud Detector — use Red Team to iterate quickly; Inspector to explain. Next: add LLM agent for polished narratives.")
