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
# CONSTANTS — per-equipment configs
# ─────────────────────────────────────────────

# AHU / Chiller features (AI4I dataset)
HVAC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_L",
    "Type_M",
]

# Elevator features (elevator dataset)
ELEV_FEATURES = [
    "revolutions",
    "humidity",
    "vibration",
    "x1", "x2", "x3", "x4", "x5",
]

# Aliases used generically in agent logic
FEATURE_NAMES = HVAC_FEATURES   # overridden per equipment at runtime

HVAC_TARGET = "Machine failure"
ELEV_TARGET = "failure"
DB_PATH     = "fm_feedback.db"

# Repair cost estimates per equipment type (SGD)
REPAIR_COSTS = {
    "AHU":      {"LOW": 500,   "MEDIUM": 2_000,  "HIGH": 8_000},
    "Chiller":  {"LOW": 1_000, "MEDIUM": 5_000,  "HIGH": 20_000},
    "Elevator": {"LOW": 300,   "MEDIUM": 1_500,  "HIGH": 6_000},
}

# Named thresholds — per equipment
THRESHOLDS_HVAC = {
    "Air temperature [K]":      {"label": "High air temperature",     "op": ">", "value": 302},
    "Process temperature [K]":  {"label": "High process temperature", "op": ">", "value": 310},
    "Rotational speed [rpm]":   {"label": "High rotational speed",    "op": ">", "value": 1_600},
    "Torque [Nm]":              {"label": "High mechanical torque",   "op": ">", "value": 50},
    "Tool wear [min]":          {"label": "Equipment aging",          "op": ">", "value": 200},
}

THRESHOLDS_ELEV = {
    "vibration":   {"label": "High vibration",          "op": ">", "value": 39.21},
    "x4":          {"label": "High motor load (x4)",    "op": ">", "value": 5164.0},
    "revolutions": {"label": "Low shaft speed",         "op": "<", "value": 21.46},
    "humidity":    {"label": "High cabin humidity",     "op": ">", "value": 75.12},
    "x1":          {"label": "Abnormal signal x1",      "op": ">", "value": 142.98},
    "x5":          {"label": "Low baseline signal x5",  "op": "<", "value": 5373.62},
}

def get_thresholds(equipment: str) -> dict:
    return THRESHOLDS_ELEV if equipment == "Elevator" else THRESHOLDS_HVAC

def get_feature_names(equipment: str) -> list:
    return ELEV_FEATURES if equipment == "Elevator" else HVAC_FEATURES

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("ai4i2020.csv")
    df = df.drop(columns=["UDI", "Product ID"])
    df = pd.get_dummies(df, columns=["Type"], drop_first=True)
    for col in ["Type_L", "Type_M"]:
        if col not in df.columns:
            df[col] = 0
    return df


@st.cache_data
def load_elevator_data() -> pd.DataFrame:
    df = pd.read_csv("elevator.csv")
    # Drop ID if present (not a feature)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    # Fill missing vibration values with median
    if df["vibration"].isnull().any():
        df["vibration"] = df["vibration"].fillna(df["vibration"].median())
    return df


def make_equipment_dataset(df: pd.DataFrame, equipment: str) -> pd.DataFrame:
    eq_df = df.copy()
    if equipment == "Chiller":
        eq_df["Process temperature [K]"] += 4.5
        eq_df["Rotational speed [rpm]"]  *= 0.88
        eq_df["Torque [Nm]"]             *= 1.15
    return eq_df

# ─────────────────────────────────────────────
# 2. PER-EQUIPMENT MODEL TRAINING
# ─────────────────────────────────────────────

@st.cache_resource
def train_models(df: pd.DataFrame) -> dict:
    """Train calibrated RandomForest for AHU and Chiller."""
    models = {}
    reports = {}

    for equipment in ["AHU", "Chiller"]:
        eq_df = make_equipment_dataset(df, equipment)
        X = eq_df[HVAC_FEATURES]
        y = eq_df[HVAC_TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        base_clf = RandomForestClassifier(
            n_estimators=150, max_depth=12,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
        clf.fit(X_train, y_train)
        reports[equipment] = classification_report(X_test.assign(y=y_test).pipe(
            lambda d: (clf.predict(d[HVAC_FEATURES]), d["y"])
        )[1], clf.predict(X_test), output_dict=True)
        models[equipment] = clf

    return models, reports


@st.cache_resource
def train_elevator_model(elev_df: pd.DataFrame):
    """Train calibrated RandomForest for Elevator using real sensor data."""
    X = elev_df[ELEV_FEATURES]
    y = elev_df[ELEV_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    base_clf = RandomForestClassifier(
        n_estimators=150, max_depth=12,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
    clf.fit(X_train, y_train)
    y_pred  = clf.predict(X_test)
    report  = classification_report(y_test, y_pred, output_dict=True)
    return clf, X_test, report




@st.cache_data
def get_training_data(df: pd.DataFrame, equipment: str) -> pd.DataFrame:
    feat = get_feature_names(equipment)
    return make_equipment_dataset(df, equipment)[feat]

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

def diagnose(features: dict, equipment: str = "AHU") -> list[str]:
    """Named-feature threshold checks — equipment-aware."""
    issues = []
    thresholds = get_thresholds(equipment)
    for feat, cfg in thresholds.items():
        val = features.get(feat, 0)
        if cfg["op"] == ">" and val > cfg["value"]:
            issues.append(f"{cfg['label']} ({feat}: {val:.2f} > {cfg['value']})")
        elif cfg["op"] == "<" and val < cfg["value"]:
            issues.append(f"{cfg['label']} ({feat}: {val:.2f} < {cfg['value']})")
    return issues


def predict(model, features: dict, equipment: str = "AHU") -> tuple[float, float, str]:
    """
    Returns (ml_prob, blended_prob, risk_label).
    Blending: 65% ML + 35% rule signal from issue count.
    Equipment-aware: uses correct feature list and thresholds.
    """
    feat_names   = get_feature_names(equipment)
    thresholds   = get_thresholds(equipment)
    X            = pd.DataFrame([features])[feat_names]
    ml_prob      = model.predict_proba(X)[0][1]

    n_issues     = len(diagnose(features, equipment))
    max_issues   = len(thresholds)
    issue_signal = n_issues / max_issues if max_issues else 0

    blended = round(0.65 * ml_prob + 0.35 * issue_signal, 4)

    if blended > 0.55:
        risk = "HIGH"
    elif blended > 0.28:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return round(ml_prob, 4), blended, risk


def recommend(risk: str, issues: list[str]) -> str:
    n = len(issues)
    if risk == "HIGH":
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

def sensor_replay(df: pd.DataFrame, elev_df: pd.DataFrame, equipment: str):
    """
    Yields one row at a time from the correct dataset, simulating a live
    sensor feed. In production, replace with an MQTT subscriber or BMS REST call.
    """
    feat = get_feature_names(equipment)
    if equipment == "Elevator":
        source = elev_df[feat].dropna()
    else:
        source = make_equipment_dataset(df, equipment)[feat]
    samples = source.sample(frac=1, random_state=int(__import__("time").time()) % 9999).reset_index(drop=True)
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
    df       = load_data()
    elev_df  = load_elevator_data()
    models, reports           = train_models(df)
    elev_model, elev_X_test, elev_report = train_elevator_model(elev_df)

    # ── Sidebar: equipment + sensor inputs ──────────────────────
    st.sidebar.header("⚙️ Configuration")
    equipment = st.sidebar.selectbox("Equipment type", ["AHU", "Chiller", "Elevator"])

    # Resolve model
    if equipment == "Elevator":
        model = elev_model
    else:
        model = models[equipment]

    st.sidebar.subheader("Sensor readings")

    if equipment == "Elevator":
        features = {
            "revolutions": st.sidebar.slider("Revolutions (rpm)",  16.0,  94.0,  46.0, 0.1),
            "humidity":    st.sidebar.slider("Humidity (%)",        72.0,  76.0,  74.0, 0.1),
            "vibration":   st.sidebar.slider("Vibration",           2.0,  100.0,  18.0, 0.1),
            "x1":          st.sidebar.slider("Signal x1",          90.0,  168.0, 120.0, 0.1),
            "x2":          st.sidebar.slider("Signal x2",         -57.0,   20.0, -28.0, 0.1),
            "x3":          st.sidebar.slider("Signal x3",           0.23,   1.27,  0.62, 0.01),
            "x4":          st.sidebar.slider("Motor load x4",     286.0, 8788.0, 2504.0, 1.0),
            "x5":          st.sidebar.slider("Baseline x5",      5241.0, 5686.0, 5510.0, 1.0),
        }
    else:
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Prediction", "📡 Live simulation", "📋 Work orders",
        "🔁 Feedback log", "📊 Model info", "🧪 AutoML — custom dataset",
    ])

    # ── TAB 1: Manual prediction ─────────────────────────────────
    with tab1:
        if "last_row_id" not in st.session_state:
            st.session_state.last_row_id = None
        if "last_feedback_given" not in st.session_state:
            st.session_state.last_feedback_given = False

        if st.button("▶ Run prediction", type="primary"):
            ml_prob, blended, risk = predict(model, features, equipment)
            issues  = diagnose(features, equipment)
            action  = recommend(risk, issues)
            row_id  = log_prediction(equipment, features, blended, risk)
            wo      = build_work_order(equipment, features, blended, risk, issues, action, row_id)

            # Persist prediction results across reruns
            st.session_state.last_row_id          = row_id
            st.session_state.last_feedback_given  = False
            st.session_state.last_ml_prob         = ml_prob
            st.session_state.last_blended         = blended
            st.session_state.last_risk            = risk
            st.session_state.last_issues          = issues
            st.session_state.last_action          = action
            st.session_state.last_wo              = wo

        # ── Show results if a prediction has been run ────────────
        if st.session_state.last_row_id is not None:
            ml_prob = st.session_state.last_ml_prob
            blended = st.session_state.last_blended
            risk    = st.session_state.last_risk
            issues  = st.session_state.last_issues
            action  = st.session_state.last_action
            wo      = st.session_state.last_wo
            row_id  = st.session_state.last_row_id

            # ── Primary metrics ──────────────────────────────────
            col1, col2, col3 = st.columns(3)
            col1.metric("Blended risk score", f"{blended:.1%}",
                        help="65% ML model + 35% rule-based issue signal")
            col2.metric("Risk level", risk)
            col3.metric("Priority score", f"${wo['priority_score']:,.0f}")

            # ── Score breakdown ──────────────────────────────────
            with st.expander("📊 Score breakdown", expanded=True):
                bcol1, bcol2 = st.columns(2)
                bcol1.metric("ML model probability", f"{ml_prob:.1%}",
                             help="What the Random Forest learned from historical failures")
                n_iss        = len(issues)
                rule_signal  = round(n_iss / len(get_thresholds(equipment)), 3)
                bcol2.metric("Rule signal (issues)", f"{rule_signal:.1%}",
                             f"{n_iss} of {len(get_thresholds(equipment))} thresholds breached",
                             help="Fraction of sensor thresholds currently in violation")

                if ml_prob < 0.30 and n_iss >= 3:
                    st.info(
                        f"ℹ️ The ML model alone scores this low ({ml_prob:.1%}), but "
                        f"{n_iss} sensors are simultaneously in violation. "
                        "The blended score reflects both signals — this combination "
                        "warrants attention even if the model hasn't seen this exact "
                        "pattern frequently in training data."
                    )

            st.markdown(f"**Work order:** `{wo['wo_id']}`")
            st.markdown(f"**Recommendation:** {action}")

            if issues:
                st.subheader("Diagnosed issues")
                for i in issues:
                    st.warning(i)
            else:
                st.success("No threshold violations detected.")

            # ── Technician feedback ──────────────────────────────
            st.subheader("🔁 Technician feedback")

            if st.session_state.last_feedback_given:
                st.success("✅ Feedback recorded — thank you. Run a new prediction to continue.")
            else:
                st.caption(f"Recording feedback for **WO-{row_id:05d}**")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("✅ Failure confirmed", key="btn_confirm"):
                        record_feedback(row_id, confirmed=True)
                        st.session_state.last_feedback_given = True
                        st.rerun()
                with col_b:
                    if st.button("❌ False alarm", key="btn_false"):
                        record_feedback(row_id, confirmed=False)
                        st.session_state.last_feedback_given = True
                        st.rerun()

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
            eq_df = elev_df if equipment == 'Elevator' else make_equipment_dataset(df, equipment)
            feat_cols = get_feature_names(equipment)
            rows  = eq_df[feat_cols].sample(n_steps, random_state=int(time.time()) % 9999).to_dict("records")
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
                ml_prob, blended, risk = predict(eq_model, live_features, eq)
                issues        = diagnose(live_features, eq)
                action        = recommend(risk, issues)

                st.session_state.sim_history.append({
                    "step": step + 1, "prob": blended, "risk": risk, "action": action,
                    "features": live_features,
                })
                st.session_state.sim_step += 1

                # ── Live metrics ─────────────────────────────────
                st.markdown(f"### Step {step + 1} / {len(rows)} — {eq}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Blended risk score", f"{blended:.1%}")
                c2.metric("Risk level", risk)
                c3.metric("Priority score", f"${priority_score(blended, eq, risk):,.0f}")

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
                    # Log highest-risk step to DB as a work order
                    history = st.session_state.sim_history
                    worst   = max(history, key=lambda h: h["prob"])
                    sim_row_id = log_prediction(
                        eq, worst["features"], worst["prob"], worst["risk"]
                    )
                    st.session_state.sim_worst_row_id = sim_row_id
                    st.rerun()
            else:
                st.session_state.sim_running = False
                st.rerun()

        elif st.session_state.sim_history:
            history = st.session_state.sim_history
            steps_x = [h["step"] for h in history]
            probs_y = [h["prob"]  for h in history]
            colors  = []
            for h in history:
                if h["risk"] == "HIGH":     colors.append("#E24B4A")
                elif h["risk"] == "MEDIUM": colors.append("#EF9F27")
                else:                       colors.append("#1D9E75")

            # ── Trend chart ──────────────────────────────────────
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.plot(steps_x, probs_y, color="#888780", linewidth=1.2, zorder=1)
            ax.scatter(steps_x, probs_y, c=colors, s=40, zorder=2)
            ax.axhline(0.55, color="#E24B4A", linestyle="--", linewidth=0.8, label="High (0.55)")
            ax.axhline(0.28, color="#EF9F27", linestyle="--", linewidth=0.8, label="Medium (0.28)")
            ax.set_xlabel("Simulation step")
            ax.set_ylabel("Blended risk score")
            ax.set_ylim(0, 1)
            ax.set_title("Simulation result — full run", fontsize=11)
            ax.legend(fontsize=8, loc="upper left")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # ── Conclusion verdict ───────────────────────────────
            n_total  = len(history)
            n_high   = sum(1 for h in history if h["risk"] == "HIGH")
            n_medium = sum(1 for h in history if h["risk"] == "MEDIUM")
            avg_prob = sum(h["prob"] for h in history) / n_total
            worst    = max(history, key=lambda h: h["prob"])

            st.markdown("---")
            st.markdown("### 📋 Simulation conclusion")

            v_col1, v_col2, v_col3 = st.columns(3)
            v_col1.metric("Avg blended score",  f"{avg_prob:.1%}")
            v_col2.metric("High-risk steps",    f"{n_high} / {n_total}")
            v_col3.metric("Peak risk score",    f"{worst['prob']:.1%}",
                          f"Step {worst['step']} — {worst['risk']}")

            # Overall maintenance verdict
            if n_high >= 3 or avg_prob > 0.55:
                st.error(
                    "⛔ **Maintenance required — schedule immediately.**  \n"
                    f"{n_high} of {n_total} steps flagged HIGH risk. "
                    f"Peak score {worst['prob']:.1%} at step {worst['step']}. "
                    "Escalate to senior technician and raise urgent work order."
                )
            elif n_high >= 1 or n_medium >= int(n_total * 0.3) or avg_prob > 0.28:
                st.warning(
                    "⚠️ **Preventive maintenance recommended within 48 h.**  \n"
                    f"{n_high} HIGH + {n_medium} MEDIUM steps detected. "
                    f"Average blended score {avg_prob:.1%}. "
                    "Schedule inspection before next operational cycle."
                )
            else:
                st.success(
                    "✅ **Equipment operating normally — no immediate action required.**  \n"
                    f"All {n_total} steps within acceptable range. "
                    f"Average score {avg_prob:.1%}. Continue routine monitoring schedule."
                )

            # ── Step-by-step table ───────────────────────────────
            with st.expander("View step-by-step breakdown"):
                hist_df = pd.DataFrame([
                    {"Step": h["step"],
                     "Blended score": f"{h['prob']:.1%}",
                     "Risk":   h["risk"],
                     "Action": h["action"]}
                    for h in history
                ])
                st.dataframe(hist_df, use_container_width=True, hide_index=True)

    # ── TAB 3: Work order queue ───────────────────────────────────
    with tab3:
        st.subheader("Work order queue (ranked by priority score)")
        st.caption("Includes both manual predictions and simulation results.")
        fb_df = load_feedback()

        if fb_df.empty:
            st.info("No predictions logged yet. Run a prediction or simulation first.")
        else:
            import json
            wo_rows = []
            for _, row in fb_df.iterrows():
                try:
                    feat = json.loads(row["features"])
                except Exception:
                    continue
                risk  = row["risk"]
                eq    = row["equipment"]
                prob  = row["probability"]
                score = priority_score(prob, eq, risk)

                # Derive purpose from risk + issues
                issues_snap = diagnose(feat)
                n_iss = len(issues_snap)
                if risk == "HIGH" or n_iss >= 4:
                    purpose = "⛔ Immediate inspection"
                elif risk == "MEDIUM" or n_iss >= 3:
                    purpose = "⚠️ Preventive maintenance"
                elif any("aging" in i.lower() for i in issues_snap):
                    purpose = "🛠️ Scheduled overhaul"
                elif any("temperature" in i.lower() for i in issues_snap):
                    purpose = "🌡️ Cooling system check"
                elif any("torque" in i.lower() for i in issues_snap):
                    purpose = "🔧 Motor load check"
                else:
                    purpose = "📋 Routine monitoring"

                confirmed_val = row["confirmed"]
                if confirmed_val == 1:   status = "✅ Confirmed"
                elif confirmed_val == 0: status = "❌ False alarm"
                else:                    status = "⏳ Pending"

                wo_rows.append({
                    "WO ID":       f"WO-{row['id']:05d}",
                    "Timestamp":   row["ts"][:16].replace("T", " "),
                    "Equipment":   eq,
                    "Risk":        risk,
                    "Score":       f"{prob:.1%}",
                    "Priority ($)": score,
                    "Purpose":     purpose,
                    "Status":      status,
                })

            wo_df = (
                pd.DataFrame(wo_rows)
                .sort_values("Priority ($)", ascending=False)
                .reset_index(drop=True)
            )
            # Format priority for display after sorting
            wo_df["Priority ($)"] = wo_df["Priority ($)"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(wo_df, use_container_width=True, hide_index=True)

    # ── TAB 4: Feedback log ───────────────────────────────────────
    with tab4:
        st.subheader("Technician feedback log")
        fb_df = load_feedback()

        if fb_df.empty:
            st.info("No predictions logged yet — run a prediction in the Prediction tab first.")
        else:
            confirmed   = fb_df[fb_df["confirmed"] == 1]
            false_alarm = fb_df[fb_df["confirmed"] == 0]
            pending     = fb_df[fb_df["confirmed"].isna()]

            c1, c2, c3 = st.columns(3)
            c1.metric("✅ Confirmed failures", len(confirmed))
            c2.metric("❌ False alarms",       len(false_alarm))
            c3.metric("⏳ Pending review",     len(pending))

            if (len(confirmed) + len(false_alarm)) > 0:
                precision = len(confirmed) / (len(confirmed) + len(false_alarm))
                st.progress(precision, text=f"Field precision: {precision:.1%}")

            # Human-readable status column
            def status_label(row):
                if row["confirmed"] == 1:   return "✅ Confirmed failure"
                if row["confirmed"] == 0:   return "❌ False alarm"
                return "⏳ Pending"

            display_df = fb_df.copy()
            display_df["Status"]      = display_df.apply(status_label, axis=1)
            display_df["Probability"] = display_df["probability"].apply(lambda x: f"{x:.1%}")
            display_df["Timestamp"]   = display_df["ts"].str[:19].str.replace("T", " ")

            st.dataframe(
                display_df[["id", "Timestamp", "equipment", "risk", "Probability", "Status", "notes"]]
                .rename(columns={"id": "WO", "equipment": "Equipment",
                                 "risk": "Risk", "notes": "Notes"}),
                use_container_width=True,
                hide_index=True,
            )

            # Allow feedback on any pending row directly from this tab
            pending_rows = fb_df[fb_df["confirmed"].isna()]
            if not pending_rows.empty:
                st.markdown("---")
                st.markdown("**Give feedback on pending predictions:**")
                for _, row in pending_rows.iterrows():
                    with st.expander(f"WO-{row['id']:05d} | {row['equipment']} | {row['risk']} | {row['ts'][:16]}"):
                        fa, fb2 = st.columns(2)
                        with fa:
                            if st.button("✅ Confirm failure", key=f"confirm_{row['id']}"):
                                record_feedback(int(row["id"]), confirmed=True)
                                st.rerun()
                        with fb2:
                            if st.button("❌ False alarm", key=f"false_{row['id']}"):
                                record_feedback(int(row["id"]), confirmed=False)
                                st.rerun()

            if st.button("🔄 Trigger model retraining"):
                aug = retrain_from_feedback(df, equipment)
                if aug is None:
                    st.warning("Need at least 20 confirmed feedback cases to retrain. Keep collecting!")
                else:
                    st.success("Augmented dataset ready. Plug into your CI/CD retraining pipeline.")

            st.markdown("---")
            st.markdown("**Danger zone**")
            if "confirm_clear" not in st.session_state:
                st.session_state.confirm_clear = False

            if not st.session_state.confirm_clear:
                if st.button("🗑️ Clear all feedback data"):
                    st.session_state.confirm_clear = True
                    st.rerun()
            else:
                st.warning("This will permanently delete all predictions and feedback. Are you sure?")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("Yes, delete everything", type="primary"):
                        con = sqlite3.connect(DB_PATH)
                        con.execute("DELETE FROM feedback")
                        con.commit()
                        con.close()
                        st.session_state.confirm_clear = False
                        st.session_state.last_row_id = None
                        st.session_state.last_feedback_given = False
                        st.success("All data cleared.")
                        st.rerun()
                with col_no:
                    if st.button("Cancel"):
                        st.session_state.confirm_clear = False
                        st.rerun()

    # ── TAB 5: Model info ─────────────────────────────────────────
    with tab5:
        st.subheader(f"Model performance — {equipment}")
        report     = elev_report if equipment == "Elevator" else reports[equipment]
        label_key  = "1" if "1" in report else list(report.keys())[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision (failure class)", f"{report[label_key]['precision']:.2%}")
        c2.metric("Recall (failure class)",    f"{report[label_key]['recall']:.2%}")
        c3.metric("F1 (failure class)",        f"{report[label_key]['f1-score']:.2%}")

        st.subheader("Feature importance")
        base_clf = model.calibrated_classifiers_[0].estimator
        feat_df  = pd.DataFrame({
            "Feature":    get_feature_names(equipment),
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
            if equipment == "Elevator":
                X_sample = elev_X_test.sample(min(50, len(elev_X_test)), random_state=1)
            else:
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

    # ── TAB 6: AutoML — custom dataset ───────────────────────────
    with tab6:
        st.subheader("🧪 AutoML — bring your own dataset")
        st.markdown(
            "Upload any CSV or Excel file. The agent will **ingest → clean → train → deploy** "
            "a failure prediction model automatically. You can then run predictions against it."
        )

        # ── Session state ────────────────────────────────────────
        for k, v in {
            "aml_model":      None,
            "aml_features":   None,
            "aml_target":     None,
            "aml_report":     None,
            "aml_X_test":     None,
            "aml_cleaned":    None,
            "aml_clean_log":  [],
            "aml_trained":    False,
        }.items():
            if k not in st.session_state:
                st.session_state[k] = v

        # ── Step 1: Upload ───────────────────────────────────────
        st.markdown("### Step 1 — Upload dataset")
        uploaded = st.file_uploader(
            "Upload CSV or Excel (.csv, .xlsx, .xls)",
            type=["csv", "xlsx", "xls"],
            key="aml_upload"
        )

        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    raw_df = pd.read_csv(uploaded)
                else:
                    raw_df = pd.read_excel(uploaded)
                st.success(f"Loaded **{uploaded.name}** — {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")

                with st.expander("Preview raw data"):
                    st.dataframe(raw_df.head(10), use_container_width=True)

                # Stats
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Rows",    f"{raw_df.shape[0]:,}")
                sc2.metric("Columns", raw_df.shape[1])
                sc3.metric("Missing values", int(raw_df.isnull().sum().sum()))

                # ── Step 2: Configure ────────────────────────────
                st.markdown("### Step 2 — Configure")
                all_cols    = list(raw_df.columns)
                target_col  = st.selectbox(
                    "Select target column (what to predict — e.g. failure, fault, label)",
                    all_cols,
                    index=len(all_cols) - 1,
                    key="aml_target_sel"
                )

                drop_cols = st.multiselect(
                    "Drop columns (IDs, timestamps, irrelevant fields)",
                    [c for c in all_cols if c != target_col],
                    default=[c for c in all_cols if c.lower() in
                             ("id", "uid", "uuid", "index", "timestamp", "date", "time")],
                    key="aml_drop_cols"
                )

                # Show target distribution
                st.markdown(f"**Target column preview — `{target_col}`**")
                vc = raw_df[target_col].value_counts().head(10)
                fig_t, ax_t = plt.subplots(figsize=(6, 2.5))
                vc.plot(kind="bar", ax=ax_t, color="#378ADD")
                ax_t.set_title(f"'{target_col}' distribution", fontsize=10)
                ax_t.set_xlabel("")
                fig_t.tight_layout()
                st.pyplot(fig_t)
                plt.close(fig_t)

                # ── Step 3: Clean & Train ────────────────────────
                st.markdown("### Step 3 — Clean & train")
                if st.button("🚀 Run AutoML pipeline", type="primary"):
                    work_df = raw_df.drop(columns=drop_cols, errors="ignore")

                    with st.status("Running AutoML pipeline...", expanded=True) as status:
                        st.write("🔍 Ingesting and profiling data...")
                        time.sleep(0.4)

                        st.write("🧹 Cleaning: handling nulls, encoding categoricals, removing duplicates...")
                        cleaned, clean_log = automl_clean(work_df, target_col)
                        time.sleep(0.3)

                        st.write("🤖 Training calibrated Random Forest with cross-validation...")
                        model, feat_names, X_test, y_test, report = automl_train(cleaned, target_col)
                        time.sleep(0.3)

                        st.write("📊 Evaluating on held-out test set...")
                        time.sleep(0.2)

                        # Persist to session state
                        st.session_state.aml_model    = model
                        st.session_state.aml_features = feat_names
                        st.session_state.aml_target   = target_col
                        st.session_state.aml_report   = report
                        st.session_state.aml_X_test   = X_test
                        st.session_state.aml_cleaned  = cleaned
                        st.session_state.aml_clean_log = clean_log
                        st.session_state.aml_trained  = True
                        status.update(label="✅ Pipeline complete!", state="complete")

            except Exception as e:
                st.error(f"Error reading file: {e}")

        # ── Step 4: Results (shown after training) ───────────────
        if st.session_state.aml_trained and st.session_state.aml_model:
            st.markdown("---")
            st.markdown("### Step 4 — Pipeline results")

            # Cleaning log
            with st.expander("🧹 Cleaning log", expanded=False):
                for line in st.session_state.aml_clean_log:
                    st.markdown(f"- {line}")

            # Model performance
            report     = st.session_state.aml_report
            label_key  = "1" if "1" in report else list(report.keys())[0]
            st.markdown("**Model performance on test set**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precision", f"{report[label_key]['precision']:.2%}")
            m2.metric("Recall",    f"{report[label_key]['recall']:.2%}")
            m3.metric("F1 score",  f"{report[label_key]['f1-score']:.2%}")
            m4.metric("Test samples", f"{report[label_key]['support']:.0f}")

            # Feature importance chart
            base_clf  = st.session_state.aml_model.calibrated_classifiers_[0].estimator
            feat_imp  = pd.DataFrame({
                "Feature":    st.session_state.aml_features,
                "Importance": base_clf.feature_importances_,
            }).sort_values("Importance", ascending=False).head(15)

            fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
            sns.barplot(data=feat_imp, x="Importance", y="Feature",
                        ax=ax_imp, palette="flare", orient="h")
            ax_imp.set_title("Top feature importances (custom model)", fontsize=11)
            fig_imp.tight_layout()
            st.pyplot(fig_imp)
            plt.close(fig_imp)

            # ── Step 5: Run predictions ──────────────────────────
            st.markdown("---")
            st.markdown("### Step 5 — Predict on new data")
            st.markdown(
                "Enter values for each feature below, or paste a single CSV row. "
                "The deployed custom model will score it instantly."
            )

            feat_names  = st.session_state.aml_features
            cleaned_df  = st.session_state.aml_cleaned
            input_vals  = {}

            # Auto-generate sliders / inputs from feature stats
            cols_per_row = 3
            feat_chunks  = [feat_names[i:i+cols_per_row]
                            for i in range(0, len(feat_names), cols_per_row)]
            for chunk in feat_chunks:
                row_cols = st.columns(len(chunk))
                for col_w, feat in zip(row_cols, chunk):
                    with col_w:
                        series = cleaned_df[feat] if feat in cleaned_df.columns else None
                        if series is not None and pd.api.types.is_numeric_dtype(series):
                            mn  = float(series.min())
                            mx  = float(series.max())
                            med = float(series.median())
                            # Binary dummy columns → checkbox
                            if set(series.unique()).issubset({0, 1}):
                                input_vals[feat] = int(st.checkbox(feat, key=f"aml_inp_{feat}"))
                            else:
                                input_vals[feat] = st.number_input(
                                    feat, min_value=mn, max_value=mx,
                                    value=med, key=f"aml_inp_{feat}"
                                )
                        else:
                            input_vals[feat] = st.number_input(feat, value=0.0, key=f"aml_inp_{feat}")

            if st.button("🔍 Predict with custom model", type="primary", key="aml_predict"):
                aml_model = st.session_state.aml_model
                X_input   = pd.DataFrame([input_vals])[feat_names]
                prob      = aml_model.predict_proba(X_input)[0][1]

                if prob > 0.60:   risk = "HIGH";   color = "error"
                elif prob > 0.30: risk = "MEDIUM";  color = "warning"
                else:             risk = "LOW";     color = "success"

                r1, r2 = st.columns(2)
                r1.metric("Failure probability (custom model)", f"{prob:.1%}")
                r2.metric("Risk level", risk)

                getattr(st, color)(
                    {"HIGH":   "⛔ Immediate inspection recommended",
                     "MEDIUM": "⚠️ Schedule preventive maintenance within 48 h",
                     "LOW":    "✅ Normal — continue monitoring"}[risk]
                )

            # Download cleaned dataset
            st.markdown("---")
            csv_bytes = st.session_state.aml_cleaned.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download cleaned dataset",
                data=csv_bytes,
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )



# ─────────────────────────────────────────────
# 8. AUTOML PIPELINE HELPERS
# ─────────────────────────────────────────────

def automl_clean(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Auto-clean any uploaded dataset:
    1. Drop columns with >50% missing
    2. Fill numeric NaN with median
    3. Fill categorical NaN with mode
    4. One-hot encode categoricals
    5. Drop duplicate rows
    Returns cleaned df + a log of actions taken.
    """
    log   = []
    orig  = df.shape

    # Drop high-null columns (excluding target)
    null_frac = df.drop(columns=[target_col]).isnull().mean()
    drop_cols = null_frac[null_frac > 0.5].index.tolist()
    if drop_cols:
        df  = df.drop(columns=drop_cols)
        log.append(f"Dropped {len(drop_cols)} column(s) with >50% missing: {drop_cols}")

    # Separate features / target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Fill numeric
    num_cols = X.select_dtypes(include="number").columns
    for col in num_cols:
        n_null = X[col].isnull().sum()
        if n_null:
            X[col] = X[col].fillna(X[col].median())
            log.append(f"Filled {n_null} missing values in '{col}' with median")

    # Fill & encode categoricals
    cat_cols = X.select_dtypes(exclude="number").columns
    for col in cat_cols:
        n_null = X[col].isnull().sum()
        if n_null:
            X[col] = X[col].fillna(X[col].mode()[0])
            log.append(f"Filled {n_null} missing values in '{col}' with mode")
    if len(cat_cols):
        X   = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        log.append(f"One-hot encoded {len(cat_cols)} categorical column(s)")

    # Drop duplicates
    before = len(X)
    mask   = ~X.duplicated()
    X, y   = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    dupes  = before - len(X)
    if dupes:
        log.append(f"Removed {dupes} duplicate rows")

    cleaned = pd.concat([X, y.reset_index(drop=True)], axis=1)
    log.insert(0, f"Shape: {orig} → {cleaned.shape}")
    return cleaned, log


def automl_train(cleaned: pd.DataFrame, target_col: str) -> tuple:
    """
    Train a calibrated RandomForest on any cleaned dataset.
    Returns (model, feature_names, X_test, y_test, report).
    """
    X = cleaned.drop(columns=[target_col])
    y = cleaned[target_col]

    # Ensure binary target
    unique_vals = y.nunique()
    if unique_vals > 10:
        median_val = y.median()
        y = (y > median_val).astype(int)

    feat_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base = RandomForestClassifier(
        n_estimators=150, max_depth=12,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, feat_names, X_test, y_test, report


if __name__ == "__main__":
    main()
