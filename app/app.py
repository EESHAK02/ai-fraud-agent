# app/app_simple_presets.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from src.config import K
from src.core.data import load_or_demo, prepare
from src.core.model import fit_if, score_if
from src.core.attack import inject_burst
from src.core.attack_simple import inject_simple_burst
from src.core.insights import build_user_profiles, compare_txn_to_user
from src.explain.report_simple import build_incident_summary, build_llm_overview
from src.explain.agent import llm_explain_row

# ---------- helpers ----------
# def _minmax(x):
#     x = np.asarray(x, float)
#     mx, mn = np.nanmax(x), np.nanmin(x)
#     return np.zeros_like(x) if mx == mn else (x - mn) / (mx - mn)

def _percentile_rank(series, value):
    """Return percentile rank (0..100) of value w.r.t. series."""
    if series is None or len(series) == 0:
        return np.nan
    return float((series < value).mean() * 100.0)

def _why_flagged(cmp):
    """Human-readable reasons from compare_txn_to_user outputs."""
    reasons = []
    # amount z-score
    if cmp.get("z_amount") is not None:
        z = abs(cmp["z_amount"])
        if z >= 3: reasons.append(f"Amount extreme (|z| ≈ {cmp['z_amount']:.1f})")
        elif z >= 2: reasons.append(f"Amount high vs user (z ≈ {cmp['z_amount']:.1f})")
    # velocity / fanout
    if cmp.get("vel_5m", 0) >= max(2, cmp.get("vel_typical_p95", 2)):
        reasons.append(f"High short-term velocity ({int(cmp['vel_5m'])} in 5m)")
    if cmp.get("fanout_5m", 0) >= max(3, cmp.get("fanout_typical_p95", 3)):
        reasons.append(f"Multiple merchants in 5m (fan-out={int(cmp['fanout_5m'])})")
    # hour rarity
    if cmp.get("hour_prob") is not None and cmp["hour_prob"] < 0.1:
        reasons.append(f"Unusual hour for this user (p≈{cmp['hour_prob']:.02f})")
    # merchant familiarity
    if not cmp.get("merchant_familiar", True):
        reasons.append("New/rare merchant for this user")
    return reasons or ["Score is high versus cohort baseline"]

st.set_page_config(page_title="Agentic Fraud Detector", layout="wide")
st.title("Agentic Financial Fraud Detector")

@st.cache_data(show_spinner=False)
def load_data():
    return load_or_demo()

@st.cache_data(show_spinner=True)
def build_feats(df):
    return prepare(df)

# ---------- UI / Presets ----------
tabs = st.tabs(["Run", "Inspect", "Report"])

with tabs[0]:
    st.subheader("Run")
    colA, colB = st.columns([1, 2])

    with colB:
        st.caption("Attack presets")
        preset = st.selectbox(
            "Choose attack preset",
            [
                "No Attack",
                "Simple Large-Amount (cash-out spike)",     # uses inject_simple_burst with large amount
                "Card-testing (small repeated amounts)",    # small repeated low amounts
                "$49.99 Burst (behavioral demo, light)",   # your original burst
                "Ramp-up (amount steps)"
            ],
            index=0,
        )

        st.markdown("**Model feature mode**")
        feature_mode = st.radio(
            "Which features should the model use?",
            ("Amount-only", "Behavioral (+Amount)", "Behavioral + PCA (V* columns)"),
            index=1,
        )

        # --- Detector choice ---
        detector = st.radio(
            "Detector",
            ("Isolation Forest (unsupervised)", "XGBoost (supervised)"),
            index=0,
            help="Both use the SAME eval split. IF trains on held-out normals; XGB trains on labeled Kaggle fraud vs normal (pre-split) and ignores injected rows."
        )

        # analyst capacity
        Ksel = st.slider("Analyst capacity K (top-K alerts)", 50, 1000, K, 50)
        run_btn = st.button("Run pipeline", use_container_width=True)

    with colA:
        st.markdown("Tips:")
        st.write("- Use **Amount-only** to stress-test simple detectors (easy to see effect).")
        st.write("- Use **Behavioral** to show how velocity, fanout and z-score improve detection of stealthy attacks.")

    # ---------- Run logic ----------
    if run_btn:
        base = load_data()
        df = base.copy()
        meta = {}

        # Choose injection based on preset
        if preset == "Simple Large-Amount (cash-out spike)":
            df, meta = inject_simple_burst(
                df,
                n=5,                     # a few large cash-out txns
                amount=8000.0,
                adv_user="adv_user_cashout",
                merchants_pool=4,
                interval_s=10,
                t0_offset_minutes=10,
                clone_template=True
            )

        elif preset == "Card-testing (small repeated amounts)":
            # small repeated (e.g., $4.99) across multiple merchants -> classic card test
            df, meta = inject_simple_burst(
                df,
                n=150,
                amount=50,
                adv_user="adv_user_cardtest",
                merchants_pool=20,
                interval_s=2,
                t0_offset_minutes=10,
                clone_template=True
            )

        elif preset == "$49.99 Burst (behavioral demo, light)":
            df, meta = inject_burst(
                df, n=150, amount=49.99, interval_s=2, merchants_pool=50,
                adv_user="adv_user_0", mode="perturb_v", jitter_time_s=1.0,
                amount_jitter_pct=0.01, v_noise_scale=0.05, distributed_users=False,
                users_pool_size=1, force_label_class=None
            )

        elif preset == "Ramp-up (amount steps)":
            steps = (50, 200, 500, 1000, 2000)
            # create the ramp by calling inject_simple_burst in small chunks (or custom helper)
            # we'll create them as separate injections so meta shows t0 of first injection
            start_meta = None
            idx = 0
            rows = []
            if "timestamp" in df.columns:
                start = pd.to_datetime(df["timestamp"].min()) + pd.Timedelta(minutes=10)
            else:
                start = pd.Timestamp("2013-01-01 00:00:00") + pd.Timedelta(minutes=10)

            uid = (df["user_id"].sample(1, random_state=42).iloc[0]) if "user_id" in df.columns else "adv_user_ramp"
            for k, amt in enumerate(steps):
                ts = start + pd.Timedelta(seconds=k * 15)
                rows.append({
                    "txn_id": f"ramp_{k}_{np.random.randint(1_000_000)}",
                    "timestamp": ts,
                    "user_id": uid,
                    "merchant_id": f"m_ramp_{k}",
                    "Amount": float(amt),
                    "Class": 1,
                    "is_attack": 1
                })
            inj = pd.DataFrame(rows)
            df = pd.concat([df, inj], ignore_index=True, sort=False)
            meta = {"t0": start, "n_injected": len(rows)}

        # ---------- ensure attack flag and build features ----------
        if "is_attack" not in df.columns:
            df["is_attack"] = 0
        df["is_attack"] = df["is_attack"].fillna(0).astype(int)

        feats = build_feats(df)  # builds vel_5m, fanout_5m, z_amount, hour, etc.

        # Map raw is_attack onto feats robustly by txn_id
        label_map = dict(zip(df["txn_id"], df["is_attack"])) if "txn_id" in df.columns else None
        if "is_attack" in feats.columns:
            feats["is_attack"] = feats["is_attack"].fillna(0).astype(int)
        elif ("txn_id" in feats.columns) and (label_map is not None):
            feats["is_attack"] = feats["txn_id"].map(label_map).fillna(0).astype(int)
        elif len(feats) == len(df):
            feats["is_attack"] = df["is_attack"].to_numpy()
        else:
            feats["is_attack"] = 0

        # ---------- choose feature set for model ----------
        # Amount-only mode -> only Amount is used (fast, easy to interpret)
        # Behavioral -> Amount + vel_5m, same_amt_5m, fanout_5m, z_amount, hour
        # Behavioral+PCA -> include V* columns (if present) to leverage PCA cols from original Kaggle
        if feature_mode == "Amount-only":
            feature_cols = ["Amount"]
        else:
            feature_cols = ["Amount", "hour", "vel_5m", "same_amt_5m", "fanout_5m", "z_amount"]
            if feature_mode == "Behavioral + PCA":
                vcols = [c for c in feats.columns if str(c).upper().startswith("V")]
                feature_cols += vcols

        # safe fill
        X = feats[feature_cols].fillna(0.0)

        # ---------- training mask: unsupervised setup ----------
        is_attack = feats.get("is_attack", 0).astype(int)
        is_kaggle_fraud = (feats.get("Class", 0) == 1).astype(int)
        normal_mask = (is_attack == 0) & (is_kaggle_fraud == 0)
        t_split_normals = feats.loc[normal_mask, "timestamp"].quantile(0.8)
        # Train ONLY on presumed-normal rows:
        #train_mask = (feats["is_attack"] == 0) & (feats.get("Class", 0) == 0)
        train_mask_un = normal_mask & (feats["timestamp"] < t_split_normals)

        train_mask_sup   = (feats["timestamp"] < t_split_normals) & (is_attack == 0)

        # If we injected at t0, exclude injected timestamps from training (already in your code)
        if meta and meta.get("t0") is not None:
            train_mask = train_mask_un & (feats["timestamp"] < meta["t0"])

        # --- Define a default eval split, independent of injection ---
        # Use last 20% of time as eval; then make sure eval and train don't overlap
        # t_split = feats["timestamp"].quantile(0.8)
        # eval_mask = (feats["timestamp"] >= t_split) & (~train_mask)
        eval_mask  = ((normal_mask & (feats["timestamp"] >= t_split_normals))
              | (is_attack == 1)
              | (is_kaggle_fraud == 1))

        # Persist the masks as columns so they stay aligned after sorting
        # feats["is_train"] = train_mask
        # feats["is_eval"]  = eval_mask
        feats["is_train_unsup"] = train_mask_un
        feats["is_train_sup"]   = train_mask_sup
        feats["is_eval"]        = eval_mask

        if int(train_mask_un.sum()) < 50 or train_mask_sup.sum() < 50:
            st.warning(f"Small train set). IF may perform poorly.")

        # # scale
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # #X_train = scaler.fit_transform(X.loc[train_mask])
        # X_train  = scaler.fit_transform(X.loc[feats["is_train"]])
        # X_all = scaler.transform(X)

        # # ---------- Isolation Forest (model-only) ----------
        # from sklearn.ensemble import IsolationForest
        # iso = IsolationForest(
        #     n_estimators=300,
        #     max_samples=min(20000, max(256, X_train.shape[0])),
        #     contamination=0.02,
        #     random_state=42,
        #     n_jobs=-1
        # )
        # iso.fit(X_train)

        # # score_samples: higher = normal; negate to make "higher = anomalous"
        # scores = -iso.score_samples(X_all)
        # feats["anomaly_score"]  = scores
        # feats["combined_score"] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

        if detector.startswith("Isolation Forest"):
            X_train = scaler.fit_transform(X.loc[feats["is_train_unsup"]])
            X_all   = scaler.transform(X)

            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(
                n_estimators=300,
                max_samples=min(20000, max(256, X_train.shape[0])),
                contamination=0.02,
                random_state=42,
                n_jobs=-1
            )
            iso.fit(X_train)

            # IF: higher score = more normal → negate so higher = more anomalous
            scores = -iso.score_samples(X_all)

        else:
            # --------- XGBoost (supervised) ---------
            try:
                from xgboost import XGBClassifier
            except Exception as e:
                st.error("xgboost is not installed. `pip install xgboost`")
                raise

            X_train = scaler.fit_transform(X.loc[feats["is_train_sup"]])
            X_all   = scaler.transform(X)

            y_train = is_kaggle_fraud.loc[feats["is_train_sup"]].astype(int).to_numpy()  # 1=fraud, 0=normal
            pos = int(y_train.sum())
            neg = int((y_train == 0).sum())
            scale_pos = (neg / max(1, pos)) if pos > 0 else 1.0

            xgb = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                random_state=42, n_jobs=-1, eval_metric="logloss"
            )
            xgb.fit(X_train, y_train)

            # XGB: use fraud probability; already "higher = more anomalous"
            scores = xgb.predict_proba(X_all)[:, 1]

        # Common post-processing
        feats["model"]          = detector
        feats["anomaly_score"]  = scores
        smin, smax = float(np.min(scores)), float(np.max(scores))
        feats["combined_score"] = (scores - smin) / (smax - smin + 1e-9)
       
       
        # Keep the mask columns so they align post-sort
        scored = (feats
                .copy()
                .sort_values("combined_score", ascending=False)
                .reset_index(drop=True))

        atk_mask = (scored["is_attack"] == 1)
        injected = int(atk_mask.sum())
        topk_df = scored.head(Ksel)
        caught = int(topk_df["is_attack"].sum())

        first_rank = (int(scored[atk_mask].index.min()) + 1) if (injected > 0 and atk_mask.any()) else None

        c1, c2, c3 = st.columns(3)
        c1.metric("Injected", injected)
        c2.metric(f"In Top-"+str(Ksel), caught)
        c3.metric("First attack rank", first_rank if first_rank else "-")

        #st.line_chart(scored.sort_values("timestamp").set_index("timestamp")[["combined_score"]])


        st.session_state["scored"] = scored
        st.session_state["K"] = Ksel

with tabs[1]:
    st.subheader("Inspect")

    scored = st.session_state.get("scored")
    if scored is None or scored.empty:
        st.info("Run the pipeline first.")
    else:
        # --- Top alerts table (make selection from top N) ---
        show_cols = ["txn_id","timestamp","user_id","merchant_id","Amount","combined_score","Class"]
        topN = st.slider("How many top alerts to list here", 100, 1000, 500, 100)
        view = (scored
                .nlargest(topN, "combined_score")
                [show_cols]
                .reset_index(drop=True))
        st.dataframe(view, use_container_width=True, height=280)

        # pick a transaction to inspect
        if len(view) == 0:
            st.warning("No alerts to inspect.")
            st.stop()

        pick = st.selectbox("Choose a txn", view["txn_id"].tolist(), index=0)
        row  = scored.loc[scored["txn_id"] == pick].iloc[0]
        profiles = build_user_profiles(scored)     # your existing helper
        cmp = compare_txn_to_user(row, profiles)   # returns amount/z, vel, fanout, hour, etc.

        # --- Alert summary -----------------------------------------------------
        # percentile rank of this txn's score within entire dataset
        pct = 100.0 - _percentile_rank(scored["combined_score"].to_numpy(), row["combined_score"])
        rank = int((scored["combined_score"] > row["combined_score"]).sum() + 1)

        st.markdown("### 🔎 Alert summary")
        s1, s2, s3, s4 = st.columns([1.2,1,1,1])
        with s1:
            st.metric("Txn ID", str(row["txn_id"]))
            st.caption(f"User **{row['user_id']}** · Merchant **{row['merchant_id']}**")
        with s2:
            st.metric("Anomaly score", f"{row['combined_score']:.4f}")
            st.caption(f"Rank **#{rank}** of {len(scored):,}")
        with s3:
            label = "High" if pct >= 99 else ("Elevated" if pct >= 95 else "Moderate")
            st.metric("Model confidence", label, f"{pct:.1f}th pct")
        with s4:
            # if you keep Class in data this shows Kaggle label when present
            true_lbl = int(row.get("Class", 0))
            st.metric("Label", "Fraud" if true_lbl==1 else "Legit")

        st.markdown("---")

        # --- Metric cards (color-coded deltas in plain text) -------------------
        k1, k2, k3, k4 = st.columns(4)
        # Amount + z
        ztxt = f"z={cmp['z_amount']:.2f}" if cmp.get("z_amount") is not None else "z=n/a"
        k1.metric("Amount", f"${cmp['amount']:.2f}", ztxt)
        # Velocity
        k2.metric("Velocity (5m)", f"{int(cmp.get('vel_5m', 0))}",
                  cmp.get("vel_vs_typical", "vs typical n/a"))
        # Fan-out
        k3.metric("Fan-out (5m)", f"{int(cmp.get('fanout_5m', 0))}",
                  cmp.get("fanout_vs_typical", "vs typical n/a"))
        # Hour novelty
        if cmp.get("hour_prob") is not None:
            k4.metric("Hour novelty", f"h={int(cmp.get('hour', -1))}",
                      f"P(user|hour)={cmp['hour_prob']:.02f}")
        else:
            k4.metric("Hour novelty", f"h={int(cmp.get('hour', -1))}", "P=n/a")
        st.caption(f"Merchant familiarity: **{'known' if cmp.get('merchant_familiar', True) else 'new/rare'}** for this user")


        st.markdown("### Why was this flagged?")
        reasons = _why_flagged(cmp)
        st.write(llm_explain_row(row.to_dict()))
        for r in reasons[:5]:
            st.write(f"- {r}")
        if len(reasons) > 5:
            st.write(f"- (+{len(reasons)-5} more)")

        # Optional: show raw row for debugging
        with st.expander("Raw row (debug)"):
            st.write(row.to_frame().T)


with tabs[2]:
    st.subheader("Report")
    scored = st.session_state.get("scored")
    if scored is None or scored.empty:
        st.info("Run the pipeline first.")
    else:
        use_llm_report = st.toggle("Use LLM in summary", value=True)
        model_name = st.text_input("LLM model (Ollama)", "phi3:mini", disabled=not use_llm_report)
        #Ksel = st.slider("Top-K alerts (report)", min_value=50, max_value=500, value=200, step=50)

    
        eval_scored = scored[scored["is_eval"].astype(bool)].copy()
        n_all = len(eval_scored)
        if n_all == 0:
            st.warning("Eval slice is empty — nothing to evaluate.")
        else:
            # Effective K (cannot exceed eval rows)
            K_eff = int(min(Ksel, n_all))

            y_attack = (eval_scored.get("is_attack", 0).astype(int))  

            # Labels: POS = Kaggle fraud OR injected attack
            y_eval = (
                (eval_scored.get("Class", 0).astype(int) == 1) |
                (eval_scored.get("is_attack", 0).astype(int) == 1)
            ).astype(int).to_numpy()

            # Scores
            s_eval = eval_scored["combined_score"].to_numpy()

            # Top-K on eval slice
            top = eval_scored.nlargest(K_eff, "combined_score")

            hits_attack = int(top.get("is_attack", pd.Series(0, index=top.index)).astype(int).sum())
            prec_k_attack = hits_attack / max(1, len(top))                      # denom = actual rows in Top-K
            rec_k_attack  = hits_attack / max(1, int(y_attack.sum()))

            # Count positives in Top-K (consistent with y_eval definition)
            hits_top = int((
                (top.get("Class", 0).astype(int) == 1) |
                (top.get("is_attack", 0).astype(int) == 1)
            ).sum())

            total_pos = int(y_eval.sum())
            thr = float(top["combined_score"].iloc[-1]) if K_eff > 0 else float("nan")

            precision_k = hits_top / max(1, len(top))          # denom = actual rows in Top-K
            recall_k    = hits_top / max(1, total_pos)         # denom = all positives in eval

            m1, m2, m4, m5, m6 = st.columns(5)
            m1.metric("Eval rows", f"{n_all:,}")
            m2.metric(f"Top-{K_eff} threshold", f"{thr:.3f}" if np.isfinite(thr) else "n/a")
            m4.metric(f"In Top-{K_eff}", f"{hits_attack:,}")
            m5.metric(f"Recall@{K_eff}", f"{rec_k_attack:.3f}")
            m6.metric(f"Precision@{K_eff}", f"{prec_k_attack:.3f}")

            # ---- Overall eval (ROC/PR) ----
            from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, precision_score,recall_score, f1_score

            with st.expander("Overall evaluation (out-of-sample)"):
                st.caption("Computed on the eval slice (held-out normals + all attacks / Kaggle fraud).")
                n_pos = y_eval.sum()
                n_neg = len(y_eval) - n_pos
                if n_pos > 0 and n_neg > 0:
                    # Guard against NaNs/const scores
                    if np.allclose(s_eval.max(), s_eval.min()) or np.isnan(s_eval).any():
                        st.info("Scores are constant/NaN on eval — cannot compute AUC/PR.")
                    else:
                        # roc = roc_auc_score(y_eval, s_eval)
                        # ap  = average_precision_score(y_eval, s_eval)
                        # c1, c2 = st.columns(2)
                        # c1.metric("ROC-AUC (eval)", f"{roc:.3f}")
                        # c2.metric("PR-AUC (eval)",  f"{ap:.3f}")
                        thresh = np.percentile(s_eval, 95)      # adjust as you wish
                        y_pred = (s_eval >= thresh).astype(int)

                        # --- metrics ---
                        roc = roc_auc_score(y_eval, s_eval)
                        pr  = average_precision_score(y_eval, s_eval)
                        prec = precision_score(y_eval, y_pred, zero_division=0)
                        rec  = recall_score(y_eval, y_pred, zero_division=0)
                        f1   = f1_score(y_eval, y_pred, zero_division=0)

                        m1, m2, m3, c1, c2 = st.columns(5)
                        m1.metric("ROC-AUC",  f"{roc:.3f}")
                        m2.metric("PR-AUC",   f"{pr:.3f}")
                        c1.metric("Precision", f"{prec:.3f}")
                        c2.metric("Recall",    f"{rec:.3f}")
                        m3.metric("F1",       f"{f1:.3f}")
                        

                        
                else:
                    st.info("Eval slice needs both positives and negatives to compute AUC/PR.")

        st.markdown("---")
        st.markdown("### Analyst narrative (LLM)")
        overview = build_llm_overview(top, model=model_name if use_llm_report else None)
        st.write(overview["narrative"])
        if overview.get("triage"):
            st.markdown("### Suggested next steps")
            for t in overview["triage"]:
                st.write(f"- {t}")

        st.markdown("---")
        st.markdown("### Top alerts")
        show_cols = ["txn_id", "user_id", "merchant_id", "Amount", "combined_score", "Class"]
        st.dataframe(top[show_cols], use_container_width=True, height=360)

        md = build_incident_summary(scored, k=Ksel, use_llm=False)
        st.download_button("Download report.md", data=md, file_name="incident_report.md")
