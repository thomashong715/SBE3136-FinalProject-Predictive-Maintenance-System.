"""
FM Predictive Maintenance AI Agent — v2 (Production-grade)
==========================================================
Improvements over v1:
  1. Named-feature logic   — no more magic index access
  2. Per-equipment models  — AHU & Chiller trained separately
  3. Decision planner      — risk × cost priority score + ranked work orders
  4. Feedback loop         — technician verdicts logged to SQLite, periodic retraining
  5. Live sensor simulation— CSV-replay polling loop replaces manual sliders
"""

import time
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

FEATURE_NAMES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_L",
    "Type_M",
]

TARGET = "Machine failure"
DB_PATH = "fm_feedback.db"

# Repair cost estimates per equipment type (SGD)
REPAIR_COSTS = {
    "AHU":    {"LOW": 500,   "MEDIUM": 2_000,  "HIGH": 8_000},
    "Chiller":{"LOW": 1_000, "MEDIUM": 5_000,  "HIGH": 20_000},
}

# Named thresholds — no more magic indices
THRESHOLDS = {
    "Air temperature [K]":      {"label": "High air temperature",     "op": ">", "value": 302},
    "Process temperature [K]":  {"label": "High process temperature", "op": ">", "value": 310},
    "Rotational speed [rpm]":   {"label": "High rotational speed",    "op": ">", "value": 1_600},
    "Torque [Nm]":              {"label": "High mechanical torque",   "op": ">", "value": 50},
    "Tool wear [min]":          {"label": "Equipment aging",          "op": ">", "value": 200},
}

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("ai4i2020.csv")
    df = df.drop(columns=["UDI", "Product ID"])
    df = pd.get_dummies(df, columns=["Type"], drop_first=True)

    # Ensure both dummy columns always exist
    for col in ["Type_L", "Type_M"]:
        if col not in df.columns:
            df[col] = 0
    return df


def make_equipment_dataset(df: pd.DataFrame, equipment: str) -> pd.DataFrame:
    """
    Apply equipment-specific feature transformations to simulate
    distinct sensor profiles for AHU vs Chiller, instead of
    hacking raw values at prediction time.
    """
    eq_df = df.copy()
    if equipment == "Chiller":
        # Chillers run hotter and at lower RPM
        eq_df["Process temperature [K]"] += 4.5
        eq_df["Rotational speed [rpm]"]  *= 0.88
        eq_df["Torque [Nm]"]             *= 1.15
    # AHU uses the base dataset as-is
    return eq_df

# ─────────────────────────────────────────────
# 2. PER-EQUIPMENT MODEL TRAINING
# ─────────────────────────────────────────────

@st.cache_resource
def train_models(df: pd.DataFrame) -> dict:
    """
    Train a separate calibrated RandomForest for each equipment type.
    Calibration turns raw scores into reliable probabilities.
    """
    models = {}
    reports = {}

    for equipment in ["AHU", "Chiller"]:
        eq_df = make_equipment_dataset(df, equipment)
        X = eq_df[FEATURE_NAMES]
        y = eq_df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        base_clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        # Isotonic calibration for reliable probabilities
        clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        reports[equipment] = classification_report(y_test, y_pred, output_dict=True)
        models[equipment] = clf

    return models, reports


@st.cache_data
def get_training_data(df: pd.DataFrame, equipment: str) -> pd.DataFrame:
    return make_equipment_dataset(df, equipment)[FEATURE_NAMES]

# ─────────────────────────────────────────────
# 3. FEEDBACK DATABASE
# ─────────────────────────────────────────────

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT NOT NULL,
            equipment   TEXT NOT NULL,
            features    TEXT NOT NULL,   -- JSON blob
            probability REAL NOT NULL,
            risk        TEXT NOT NULL,
            confirmed   INTEGER,         -- 1=failure confirmed, 0=false alarm, NULL=pending
            notes       TEXT
        )
    """)
    con.commit()
    con.close()


def log_prediction(equipment: str, features: dict, prob: float, risk: str) -> int:
    import json
    con = sqlite3.connect(DB_PATH)
    cur = con.execute(
        """INSERT INTO feedback (ts, equipment, features, probability, risk)
           VALUES (?, ?, ?, ?, ?)""",
        (datetime.utcnow().isoformat(), equipment, json.dumps(features), prob, risk),
    )
    row_id = cur.lastrowid
    con.commit()
    con.close()
    return row_id


def record_feedback(row_id: int, confirmed: bool, notes: str = ""):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "UPDATE feedback SET confirmed=?, notes=? WHERE id=?",
        (int(confirmed), notes, row_id),
    )
    con.commit()
    con.close()


def load_feedback() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM feedback ORDER BY ts DESC", con)
    con.close()
    return df


def retrain_from_feedback(base_df: pd.DataFrame, equipment: str) -> dict | None:
    """
    Incorporate confirmed feedback cases into retraining.
    Returns updated classification report or None if insufficient data.
    """
    import json
    fb = load_feedback()
    confirmed = fb[(fb["equipment"] == equipment) & (fb["confirmed"].notna())]

    if len(confirmed) < 20:
        return None  # Not enough feedback yet

    rows = []
    for _, row in confirmed.iterrows():
        feat = json.loads(row["features"])
        feat[TARGET] = int(row["confirmed"])
        rows.append(feat)

    aug_df = pd.concat([base_df, pd.DataFrame(rows)], ignore_index=True)
    return aug_df

# ─────────────────────────────────────────────
# 4. AI AGENT CORE — Named-feature logic
# ─────────────────────────────────────────────

def diagnose(features: dict) -> list[str]:
    """Named-feature threshold checks — no magic indices."""
    issues = []
    for feat, cfg in THRESHOLDS.items():
        val = features.get(feat, 0)
        if cfg["op"] == ">" and val > cfg["value"]:
            issues.append(f"{cfg['label']} ({feat}: {val:.1f} > {cfg['value']})")
        elif cfg["op"] == "<" and val < cfg["value"]:
            issues.append(f"{cfg['label']} ({feat}: {val:.1f} < {cfg['value']})")
    return issues


def predict(model, features: dict) -> tuple[float, str]:
    """
    Returns (probability, risk_label).
    Risk combines ML probability AND simultaneous issue count:
      - 3+ violations escalates LOW → MEDIUM
      - 4+ violations escalates anything → HIGH
    Prevents the model under-calling risk when many sensors are in the red.
    """
    X = pd.DataFrame([features])[FEATURE_NAMES]
    prob = model.predict_proba(X)[0][1]

    # Base risk from model probability (lowered thresholds for FM sensitivity)
    if prob > 0.60:
        risk = "HIGH"
    elif prob > 0.30:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # Escalate based on simultaneous threshold violations
    n_issues = len(diagnose(features))
    if n_issues >= 4:
        risk = "HIGH"
    elif n_issues >= 3 and risk == "LOW":
        risk = "MEDIUM"

    return round(prob, 4), risk


def recommend(risk: str, issues: list[str]) -> str:
    n = len(issues)
    if risk == "HIGH" or n >= 4:
        return "⛔ Immediate inspection required — escalate to senior technician"
    if n >= 3:
        return "⚠️ Multiple faults detected — urgent maintenance within 24 h"
    if any("aging" in i.lower() for i in issues):
        return "🛠️ Schedule preventive maintenance within 48 h"
    if any("torque" in i.lower() for i in issues):
        return "🔧 Check motor load and belt tension"
    if any("temperature" in i.lower() for i in issues):
        return "🌡️ Inspect cooling circuit and ventilation"
    if risk == "MEDIUM":
        return "⚠️ Flag for next scheduled inspection"
    return "✅ Normal operation — continue monitoring"

# ─────────────────────────────────────────────
# 5. DECISION PLANNER
# ─────────────────────────────────────────────

def priority_score(prob: float, equipment: str, risk: str) -> float:
    """
    Priority = probability × estimated repair cost.
    This surfaces high-risk, high-cost assets at the top of the work queue.
    """
    cost = REPAIR_COSTS[equipment][risk]
    return round(prob * cost, 2)


def build_work_order(equipment: str, features: dict, prob: float, risk: str,
                     issues: list[str], action: str, row_id: int) -> dict:
    score = priority_score(prob, equipment, risk)
    return {
        "wo_id":     f"WO-{row_id:05d}",
        "equipment": equipment,
        "risk":      risk,
        "probability": f"{prob:.1%}",
        "priority_score": score,
        "issues":    "; ".join(issues) if issues else "None",
        "action":    action,
        "created":   datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
    }

# ─────────────────────────────────────────────
# 6. LIVE SENSOR SIMULATION (CSV-replay)
# ─────────────────────────────────────────────

def sensor_replay(df: pd.DataFrame, equipment: str):
    """
    Yields one row at a time from the dataset, simulating a live
    sensor feed. In production, replace this with an MQTT subscriber
    or BMS REST API call.
    """
    eq_df = make_equipment_dataset(df, equipment)
    samples = eq_df[FEATURE_NAMES].sample(frac=1, random_state=0).reset_index(drop=True)
    for _, row in samples.iterrows():
        yield row.to_dict()

# ─────────────────────────────────────────────
# 7. STREAMLIT UI
# ─────────────────────────────────────────────

def risk_badge(risk: str) -> str:
    colors = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    return f"{colors.get(risk, '⚪')} **{risk}**"


def main():
    st.set_page_config(
        page_title="FM Predictive Maintenance Agent v2",
        page_icon="🤖",
        layout="wide",
    )
    st.title("🤖 FM Predictive Maintenance AI Agent — v2")

    init_db()
    df = load_data()
    models, reports = train_models(df)

    # ── Sidebar: equipment + sensor inputs ──────────────────────
    st.sidebar.header("⚙️ Configuration")
    equipment = st.sidebar.selectbox("Equipment type", ["AHU", "Chiller"])
    model = models[equipment]

    st.sidebar.subheader("Sensor readings")
    features = {
        "Air temperature [K]":     st.sidebar.slider("Air temperature (K)",    290.0, 310.0, 298.0, 0.1),
        "Process temperature [K]": st.sidebar.slider("Process temperature (K)",300.0, 320.0, 308.0, 0.1),
        "Rotational speed [rpm]":  st.sidebar.slider("Rotational speed (rpm)", 1000,  2000,  1500),
        "Torque [Nm]":             st.sidebar.slider("Torque (Nm)",             30.0,  70.0,  40.0, 0.1),
        "Tool wear [min]":         st.sidebar.slider("Tool wear (min)",          0,     300,    50),
        "Type_L":                  int(st.sidebar.checkbox("Type L")),
        "Type_M":                  int(st.sidebar.checkbox("Type M")),
    }

    # ── Tabs ────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Prediction", "📡 Live simulation", "📋 Work orders",
        "🔁 Feedback log", "📊 Model info",
    ])

    # ── TAB 1: Manual prediction ─────────────────────────────────
    with tab1:
        if st.button("▶ Run prediction", type="primary"):
            prob, risk = predict(model, features)
            issues     = diagnose(features)
            action     = recommend(risk, issues)
            row_id     = log_prediction(equipment, features, prob, risk)
            wo         = build_work_order(equipment, features, prob, risk, issues, action, row_id)

            col1, col2, col3 = st.columns(3)
            col1.metric("Failure probability", f"{prob:.1%}")
            col2.metric("Risk level", risk)
            col3.metric("Priority score", f"${wo['priority_score']:,.0f}")

            st.markdown(f"**Work order:** `{wo['wo_id']}`")
            st.markdown(f"**Recommendation:** {action}")

            if issues:
                st.subheader("Diagnosed issues")
                for i in issues:
                    st.warning(i)
            else:
                st.success("No threshold violations detected.")

            # Technician feedback (inline)
            st.subheader("🔁 Technician feedback")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Failure confirmed"):
                    record_feedback(row_id, confirmed=True)
                    st.success("Feedback recorded — will improve next model retraining.")
            with col_b:
                if st.button("❌ False alarm"):
                    record_feedback(row_id, confirmed=False)
                    st.info("False alarm logged.")

    # ── TAB 2: Live sensor simulation ────────────────────────────
    with tab2:
        st.info(
            "Simulates a live sensor feed by replaying dataset rows. "
            "In production, swap `sensor_replay()` for an MQTT subscriber or BMS REST call."
        )

        # ── Session state init ───────────────────────────────────
        if "sim_running"      not in st.session_state: st.session_state.sim_running      = False
        if "sim_history"      not in st.session_state: st.session_state.sim_history      = []
        if "sim_step"         not in st.session_state: st.session_state.sim_step         = 0
        if "sim_replay_rows"  not in st.session_state: st.session_state.sim_replay_rows  = []
        if "sim_equipment"    not in st.session_state: st.session_state.sim_equipment    = equipment

        n_steps = st.slider("Simulation steps", 5, 50, 20, key="sim_n_steps")

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            start = st.button("▶ Start simulation", type="primary",
                              disabled=st.session_state.sim_running)
        with col_btn2:
            stop = st.button("⏹ Stop", disabled=not st.session_state.sim_running)

        if start:
            # Pre-generate all rows up front so we don't rely on a live generator
            eq_df = make_equipment_dataset(df, equipment)
            rows  = eq_df[FEATURE_NAMES].sample(n_steps, random_state=int(time.time()) % 9999).to_dict("records")
            st.session_state.sim_replay_rows  = rows
            st.session_state.sim_history      = []
            st.session_state.sim_step         = 0
            st.session_state.sim_running      = True
            st.session_state.sim_equipment    = equipment
            st.rerun()

        if stop:
            st.session_state.sim_running = False
            st.rerun()

        # ── Render current step ──────────────────────────────────
        if st.session_state.sim_running:
            step      = st.session_state.sim_step
            rows      = st.session_state.sim_replay_rows
            eq        = st.session_state.sim_equipment
            eq_model  = models[eq]

            if step < len(rows):
                live_features = rows[step]
                prob, risk    = predict(eq_model, live_features)
                issues        = diagnose(live_features)
                action        = recommend(risk, issues)

                st.session_state.sim_history.append({
                    "step": step + 1, "prob": prob, "risk": risk, "action": action,
                    "features": live_features,
                })
                st.session_state.sim_step += 1

                # ── Live metrics ─────────────────────────────────
                st.markdown(f"### Step {step + 1} / {len(rows)} — {eq}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Failure probability", f"{prob:.1%}")
                c2.metric("Risk level", risk)
                c3.metric("Priority score", f"${priority_score(prob, eq, risk):,.0f}")

                risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk, "⚪")
                st.markdown(f"**{risk_color} Action:** {action}")

                with st.expander("Sensor readings this step"):
                    st.json({k: round(v, 2) for k, v in live_features.items()})

                # ── Trend chart of all steps so far ─────────────
                history = st.session_state.sim_history
                if len(history) > 0:
                    steps_x = [h["step"] for h in history]
                    probs_y = [h["prob"]  for h in history]
                    colors  = []
                    for h in history:
                        if h["risk"] == "HIGH":   colors.append("#E24B4A")
                        elif h["risk"] == "MEDIUM": colors.append("#EF9F27")
                        else:                      colors.append("#1D9E75")

                    fig, ax = plt.subplots(figsize=(9, 3))
                    ax.plot(steps_x, probs_y, color="#888780", linewidth=1.2, zorder=1)
                    ax.scatter(steps_x, probs_y, c=colors, s=40, zorder=2)
                    ax.axhline(0.70, color="#E24B4A", linestyle="--", linewidth=0.8, label="High (0.70)")
                    ax.axhline(0.40, color="#EF9F27", linestyle="--", linewidth=0.8, label="Medium (0.40)")
                    ax.set_xlabel("Simulation step")
                    ax.set_ylabel("Failure probability")
                    ax.set_ylim(0, 1)
                    ax.set_title(f"Live failure probability — {eq}", fontsize=11)
                    ax.legend(fontsize=8, loc="upper left")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                # ── History table ────────────────────────────────
                if len(history) > 1:
                    hist_df = pd.DataFrame([
                        {"Step": h["step"], "Probability": f"{h['prob']:.1%}",
                         "Risk": h["risk"], "Action": h["action"]}
                        for h in history
                    ])
                    st.dataframe(hist_df, use_container_width=True, hide_index=True)

                # ── Auto-advance ─────────────────────────────────
                if step + 1 < len(rows):
                    time.sleep(1.2)
                    st.rerun()
                else:
                    st.session_state.sim_running = False
                    st.success(f"Simulation complete — {len(rows)} steps processed.")
            else:
                st.session_state.sim_running = False
                st.rerun()

        elif st.session_state.sim_history:
            # Show final state after simulation ends
            st.markdown("### Simulation complete")
            history = st.session_state.sim_history
            steps_x = [h["step"] for h in history]
            probs_y = [h["prob"]  for h in history]
            colors  = []
            for h in history:
                if h["risk"] == "HIGH":     colors.append("#E24B4A")
                elif h["risk"] == "MEDIUM": colors.append("#EF9F27")
                else:                       colors.append("#1D9E75")

            fig, ax = plt.subplots(figsize=(9, 3))
            ax.plot(steps_x, probs_y, color="#888780", linewidth=1.2, zorder=1)
            ax.scatter(steps_x, probs_y, c=colors, s=40, zorder=2)
            ax.axhline(0.70, color="#E24B4A", linestyle="--", linewidth=0.8, label="High (0.70)")
            ax.axhline(0.40, color="#EF9F27", linestyle="--", linewidth=0.8, label="Medium (0.40)")
            ax.set_xlabel("Simulation step")
            ax.set_ylabel("Failure probability")
            ax.set_ylim(0, 1)
            ax.set_title("Final simulation result", fontsize=11)
            ax.legend(fontsize=8, loc="upper left")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            hist_df = pd.DataFrame([
                {"Step": h["step"], "Probability": f"{h['prob']:.1%}",
                 "Risk": h["risk"], "Action": h["action"]}
                for h in history
            ])
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

    # ── TAB 3: Work order queue ───────────────────────────────────
    with tab3:
        st.subheader("Work order queue (ranked by priority score)")
        fb_df = load_feedback()

        if fb_df.empty:
            st.info("No predictions logged yet. Run a prediction in the Prediction tab first.")
        else:
            import json
            rows = []
            for _, row in fb_df.iterrows():
                try:
                    feat = json.loads(row["features"])
                except Exception:
                    continue
                risk = row["risk"]
                eq   = row["equipment"]
                prob = row["probability"]
                score = priority_score(prob, eq, risk)
                rows.append({
                    "WO ID":          f"WO-{row['id']:05d}",
                    "Timestamp":      row["ts"][:16],
                    "Equipment":      eq,
                    "Risk":           risk,
                    "Probability":    f"{prob:.1%}",
                    "Priority ($)":   f"{score:,.0f}",
                    "Confirmed":      {1: "✅ Yes", 0: "❌ No alarm"}.get(row["confirmed"], "⏳ Pending"),
                })

            wo_df = (
                pd.DataFrame(rows)
                .sort_values("Priority ($)", ascending=False, key=lambda x: x.str.replace(",", "").str.replace("$", "").astype(float))
            )
            st.dataframe(wo_df, use_container_width=True)

    # ── TAB 4: Feedback log ───────────────────────────────────────
    with tab4:
        st.subheader("Technician feedback log")
        fb_df = load_feedback()

        if fb_df.empty:
            st.info("No feedback logged yet.")
        else:
            confirmed   = fb_df[fb_df["confirmed"] == 1]
            false_alarm = fb_df[fb_df["confirmed"] == 0]
            pending     = fb_df[fb_df["confirmed"].isna()]

            c1, c2, c3 = st.columns(3)
            c1.metric("✅ Confirmed failures", len(confirmed))
            c2.metric("❌ False alarms",       len(false_alarm))
            c3.metric("⏳ Pending review",     len(pending))

            precision = (
                len(confirmed) / (len(confirmed) + len(false_alarm))
                if (len(confirmed) + len(false_alarm)) > 0
                else None
            )
            if precision is not None:
                st.progress(precision, text=f"Field precision: {precision:.1%}")

            st.dataframe(
                fb_df[["id", "ts", "equipment", "risk", "probability", "confirmed", "notes"]]
                .rename(columns={"id": "ID", "ts": "Timestamp"}),
                use_container_width=True,
            )

            if st.button("🔄 Trigger model retraining"):
                aug = retrain_from_feedback(df, equipment)
                if aug is None:
                    st.warning("Need at least 20 confirmed feedback cases to retrain. Keep collecting!")
                else:
                    st.success("Augmented dataset ready. Plug into your CI/CD retraining pipeline.")

    # ── TAB 5: Model info ─────────────────────────────────────────
    with tab5:
        st.subheader(f"Model performance — {equipment}")
        report = reports[equipment]
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision (failure class)", f"{report['1']['precision']:.2%}")
        c2.metric("Recall (failure class)",    f"{report['1']['recall']:.2%}")
        c3.metric("F1 (failure class)",        f"{report['1']['f1-score']:.2%}")

        st.subheader("Feature importance")
        base_clf = model.calibrated_classifiers_[0].estimator
        feat_df  = pd.DataFrame({
            "Feature":    FEATURE_NAMES,
            "Importance": base_clf.feature_importances_,
        }).sort_values("Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=feat_df, x="Importance", y="Feature", ax=ax,
                    palette="flare", orient="h")
        ax.set_title(f"Feature importance — {equipment}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("SHAP explainability (single sample)")
        try:
            X_sample  = get_training_data(df, equipment).sample(50, random_state=1)
            explainer = shap.TreeExplainer(base_clf)
            shap_vals = explainer.shap_values(X_sample)

            # shap_values shape varies by shap version:
            # older: list [class0_arr, class1_arr]  → pick index 1
            # newer: single 3-D array (samples, features, classes) → slice [:,:,1]
            if isinstance(shap_vals, list):
                sv = shap_vals[1]
            else:
                sv = shap_vals[:, :, 1] if shap_vals.ndim == 3 else shap_vals

            fig3, _ = plt.subplots(figsize=(8, 3))
            shap.summary_plot(sv, X_sample, show=False, plot_size=None)
            st.pyplot(fig3)
            plt.close(fig3)
        except Exception as e:
            st.warning(f"SHAP plot unavailable: {e}")

        st.subheader("Raw dataset preview")
        st.dataframe(df.head(), use_container_width=True)


if __name__ == "__main__":
    main()
