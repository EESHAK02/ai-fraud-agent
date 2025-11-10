import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
from src.config import ALPHA_DEFAULT, K
from src.core.data import load_or_demo, prepare
from src.core.model import fit_if, score_if
from src.core.rules import rule_burst_score
from src.explain.agent import llm_explain_row, is_ollama_up
from src.core.insights import build_user_profiles, compare_txn_to_user
import altair as alt
from src.explain.report_simple import build_incident_summary, build_llm_overview


st.set_page_config(page_title="Agentic Fraud Detector", layout="wide")
st.title("Agentic Financial Fraud Detector (Simplified)")

@st.cache_data(show_spinner=False)
def load_data():
    return load_or_demo()

@st.cache_data(show_spinner=True)
def build_feats(df):
    return prepare(df)

def ensure_combined(df, alpha):
    out = df.copy()
    if "anomaly_score" not in out:
        out["anomaly_score"] = 0.0
    if "rule_score" not in out:
        out["rule_score"] = rule_burst_score(out)
    def norm(x):
        x = np.asarray(x, float); mx, mn = np.nanmax(x), np.nanmin(x)
        return np.zeros_like(x) if mx==mn else (x-mn)/(mx-mn)
    out["combined_score"] = alpha*norm(out["anomaly_score"]) + (1-alpha)*norm(out["rule_score"])
    return out

tabs = st.tabs(["Run","Inspect","Report"])

with tabs[0]:
    st.subheader("Run")
    colA,colB = st.columns([2,1])
    with colB:
        st.caption("Presets")
        preset = st.radio("Attack preset", ["No Attack","$49.99 Burst (light)","$49.99 Burst (strong)"], index=0)
        alpha = st.slider("Alpha: Isolation Forest weight", 0.0, 1.0, ALPHA_DEFAULT, 0.05)
        Ksel = st.slider("Analyst capacity K", 50, 1000, K, 50)
        go = st.button("Run pipeline", use_container_width=True)

    with colA:
        if go:
            base = load_data()
            df = base.copy()
            # inject simple synthetic burst if chosen
            if "Burst" in preset:
                n = 150 if "light" in preset else 400
                t0 = df["timestamp"].min() + pd.Timedelta(minutes=10)
                user = "adv_user_0"
                merchants = [f"m_adv_{i}" for i in range(50)]
                rows = []
                for i in range(n):
                    rows.append({
                        "txn_id": f"adv_{i}",
                        "timestamp": t0 + pd.Timedelta(seconds=i),
                        "user_id": user,
                        "merchant_id": merchants[i % len(merchants)],
                        "Amount": 49.99,
                        "Class": 1,
                        "is_attack": 1,      # <-- important
                    })
                df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True, sort=False)

            # ensure is_attack exists for ALL rows (base rows = 0)
            if "is_attack" not in df.columns:
                df["is_attack"] = 0
            df["is_attack"] = df["is_attack"].fillna(0).astype(int)
            
            feats = build_feats(df)

            # Build a label map from the original df
            label_map = None
            if "txn_id" in df.columns:
                label_map = dict(zip(df["txn_id"], df["is_attack"]))

            if "is_attack" in feats.columns:
                feats["is_attack"] = feats["is_attack"].fillna(0).astype(int)
            elif ("txn_id" in feats.columns) and (label_map is not None):
                # Best path: map by txn_id
                feats["is_attack"] = feats["txn_id"].map(label_map).fillna(0).astype(int)
            elif len(feats) == len(df):
                # Last resort: assume row order preserved; copy by position
                feats["is_attack"] = df["is_attack"].to_numpy()
            else:
                # Should not happen, but prevents KeyError
                feats["is_attack"] = 0

            # train IF on non-attack (label 0) rows only if available
            feature_cols = ["Amount","hour","vel_5m","same_amt_5m","fanout_5m","z_amount"]
            X = feats[feature_cols].fillna(0)

            train_mask = (feats["is_attack"] == 0)   # train only on non-attacks
            model = fit_if(X[train_mask])

            feats["anomaly_score"] = score_if(model, X)
            feats["rule_score"] = rule_burst_score(feats)
            scored = ensure_combined(feats, alpha).sort_values("combined_score", ascending=False).reset_index(drop=True)

            # metrics (use is_attack consistently)
            atk_mask = (scored["is_attack"] == 1)
            injected = int(atk_mask.sum())

            topk_df = scored.head(Ksel)
            caught = int(topk_df["is_attack"].sum())

            if injected > 0 and atk_mask.any():
                first_idx = scored[atk_mask].index.min()
                first_rank = int(first_idx) + 1
            else:
                first_rank = None

            st.metric("Injected", injected)
            st.metric(f"In Top-{Ksel}", caught)
            st.metric("First attack rank", first_rank if first_rank else "-")

            st.line_chart(
                scored.sort_values("timestamp").set_index("timestamp")[["combined_score"]]
            )

            # persist for other tabs
            st.session_state["scored"] = scored
            st.session_state["K"] = Ksel

with tabs[1]:
    st.subheader("Inspect")
    scored = st.session_state.get("scored")
    if scored is None or scored.empty:
        st.info("Run the pipeline first.")
    else:
        profiles = build_user_profiles(scored)

        view = scored.head(200)[["txn_id","timestamp","user_id","merchant_id","Amount","combined_score","Class"]]
        st.dataframe(view)

        if view.empty:
            st.warning("No alerts to inspect.")
        else:
            pick = st.selectbox("Choose a txn", view["txn_id"].tolist())
            row = scored.loc[scored["txn_id"] == pick].iloc[0]
            cmp = compare_txn_to_user(row, profiles)

            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Amount", f"${cmp['amount']:.2f}", f"z={cmp['z_amount']:.2f}" if cmp["z_amount"] is not None else "z=n/a")
            k2.metric("Velocity (5m)", f"{cmp['vel_5m']}", cmp["vel_vs_typical"])
            k3.metric("Fan-out (5m)", f"{cmp['fanout_5m']}", cmp["fanout_vs_typical"])
            k4.metric("Hour novelty", f"h={cmp['hour']}", f"P(user|hour)={cmp['hour_prob']:.2f}" if cmp["hour_prob"] is not None else "P=n/a")
            st.caption(f"Merchant familiarity: {'known' if cmp['merchant_familiar'] else 'new/rare'} for this user")

            # Charts
            uid = row["user_id"]
            u = scored[scored["user_id"] == uid].copy()

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**User amount distribution**")
                bins = alt.Chart(u).mark_bar().encode(
                    x=alt.X("Amount:Q", bin=alt.Bin(maxbins=30)),
                    y="count()"
                )
                marker = alt.Chart(pd.DataFrame({"Amount":[row["Amount"]]})).mark_rule().encode(x="Amount:Q")
                st.altair_chart((bins + marker).properties(height=220), use_container_width=True)
            with c2:
                st.markdown("**User hour-of-day**")
                h = u.groupby("hour")["txn_id"].count().reset_index()
                bar = alt.Chart(h).mark_bar().encode(x="hour:O", y="txn_id:Q")
                marker_h = alt.Chart(pd.DataFrame({"hour":[int(row["hour"])]})).mark_rule().encode(x="hour:O")
                st.altair_chart((bar + marker_h).properties(height=220), use_container_width=True)

with tabs[2]:
    st.subheader("Report")
    scored = st.session_state.get("scored")
    if scored is None or scored.empty:
        st.info("Run the pipeline first.")
    else:
        # Inputs
        use_llm_report = st.toggle("Use LLM in summary", value=True)
        model_name = st.text_input("LLM model (Ollama)", "phi3:mini", disabled=not use_llm_report)
        K = st.slider("Top-K alerts", min_value=50, max_value=500, value=200, step=50)

        # Metrics
        top = scored.nlargest(K, "combined_score")
        n_all = len(scored)
        thr = float(top["combined_score"].iloc[-1]) if len(top) else float("nan")
        is_attack = (scored["is_attack"].fillna(0).astype(bool)
             if "is_attack" in scored else pd.Series(False, index=scored.index))

        top = scored.nlargest(K, "combined_score")
        hits = int((top["is_attack"].fillna(0).astype(int)).sum()) if "is_attack" in top else 0

        recall_k = hits / max(1, int(is_attack.sum()))
        precision_k = hits / K

        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Total scored", f"{n_all:,}")
        m2.metric(f"Top-{K} threshold", f"{thr:.3f}")
        m3.metric("Injected", f"{int(is_attack.sum()):,}")
        m4.metric(f"In Top-{K}", f"{hits:,}")
        m5.metric(f"Recall@{K}", f"{recall_k:.3f}")
        m6.metric(f"Precision@{K}", f"{precision_k:.3f}")

        st.markdown("---")

        # LLM overview block
        st.markdown("### Analyst narrative (LLM)")
        if use_llm_report:
            overview = build_llm_overview(top, model=model_name)
        else:
            overview = build_llm_overview(top, model=None)  # returns default
        st.write(overview["narrative"])
        if overview.get("triage"):
            st.markdown("### Suggested next steps")
            for t in overview["triage"]:
                st.write(f"- {t}")

        st.markdown("---")

        # Compact Top-K table + download
        st.markdown("### Top alerts")
        show_cols = ["txn_id","user_id","merchant_id","Amount","combined_score","Class"]
        st.dataframe(top[show_cols], use_container_width=True, height=360)
        # Also produce your old markdown (optional)
        md = build_incident_summary(scored, k=K, use_llm=False)
        st.download_button("Download report.md", data=md, file_name="incident_report.md")