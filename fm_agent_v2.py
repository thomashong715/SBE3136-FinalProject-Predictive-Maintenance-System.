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
import json as _json
from datetime import datetime
from pathlib import Path

try:
    from anthropic import Anthropic as _Anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

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
        feat[HVAC_TARGET] = int(row["confirmed"])
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


def predict(model, features: dict, equipment: str = "AHU",
            feat_names: list | None = None) -> tuple[float, float, str]:
    """
    Returns (ml_prob, blended_prob, risk_label).
    Blending: 65% ML + 35% rule signal from issue count.
    Equipment-aware: uses correct feature list and thresholds.
    feat_names: if provided (e.g. from automl_train), overrides the default
                feature list so the DataFrame columns match what the model
                was actually trained on.
    """
    if feat_names is None:
        feat_names = get_feature_names(equipment)
    thresholds   = get_thresholds(equipment)
    # Only select columns the model knows about; fill any missing with 0
    X = pd.DataFrame([features]).reindex(columns=feat_names, fill_value=0)
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
        page_title="FM Predictive Maintenance Agent v4",
        page_icon="⚙️",
        layout="wide",
    )

    init_db()

    # ── Sidebar ──────────────────────────────────────────────────
    st.sidebar.title("⚙️ FM Agent")
    st.sidebar.markdown("---")

    # Workflow phase indicator in sidebar
    phase = st.session_state.get("workflow_phase", 1)
    st.sidebar.markdown("**Workflow progress**")
    for i, label in enumerate(["Data ingestion", "AI analytics & training", "Prediction & decision"], start=1):
        icon = "✅" if phase > i else ("▶" if phase == i else "○")
        st.sidebar.markdown(f"{icon} Phase {i} — {label}")

    st.sidebar.markdown("---")
    equipment = st.sidebar.selectbox("Equipment type", ["AHU", "Chiller", "Elevator"],
                                     help="Select the equipment to monitor and predict failures for.")
    st.sidebar.markdown("---")
    st.sidebar.caption("FM Predictive Maintenance Agent v4")

    # ── Page header ──────────────────────────────────────────────
    st.title("⚙️ FM Predictive Maintenance Agent")
    st.caption("Three-phase workflow: data ingestion → AI analytics & training → prediction & decision")
    st.markdown("---")

    # ════════════════════════════════════════════════════════════
    # PHASE 1 — DATA INGESTION
    # ════════════════════════════════════════════════════════════
    with st.expander("📂 Phase 1 — Data ingestion", expanded=(phase == 1)):
        st.markdown("##### Step 1 · Upload equipment data")
        st.caption(
            "Upload the raw CSV for the selected equipment. The agent will auto-clean "
            "and validate the data before training begins."
        )

        # ── Upload widget ────────────────────────────────────────
        col_up, col_info = st.columns([2, 1])
        with col_up:
            uploaded = st.file_uploader(
                "Upload CSV or Excel file",
                type=["csv", "xlsx", "xls"],
                key="p1_upload",
                help="Expected columns depend on equipment. Built-in datasets load automatically if no file is uploaded.",
            )
        with col_info:
            st.markdown("**Built-in datasets**")
            st.markdown("- `ai4i2020.csv` → AHU / Chiller")
            st.markdown("- `elevator.csv` → Elevator")
            st.markdown("Upload a custom file to override.")

        # ── Load data (uploaded or built-in) ────────────────────
        df      = load_data()
        elev_df = load_elevator_data()

        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    raw_upload = pd.read_csv(uploaded)
                else:
                    raw_upload = pd.read_excel(uploaded)
                st.success(f"\u2713 Loaded `{uploaded.name}` \u2014 {raw_upload.shape[0]:,} rows \u00d7 {raw_upload.shape[1]} cols")
                st.session_state["uploaded_df"]   = raw_upload
                st.session_state["uploaded_name"] = uploaded.name
            except Exception as e:
                st.error(f"Could not read file: {e}")

        # Priority: uploaded file > built-in dataset matching equipment selection
        if "uploaded_df" in st.session_state:
            source_df = st.session_state["uploaded_df"]
            st.caption(f"Showing uploaded file: `{st.session_state.get('uploaded_name', 'custom')}`")
        else:
            source_df = df if equipment != "Elevator" else elev_df
            st.caption(f"Showing built-in dataset for **{equipment}**.")
        st.markdown("---")
        st.markdown("##### Step 2 · Data cleaning & validation")

        # ── Data quality card ────────────────────────────────────
        dq_total   = source_df.shape[0] * source_df.shape[1]
        dq_missing = int(source_df.isnull().sum().sum())
        dq_dupes   = int(source_df.duplicated().sum())
        num_cols   = source_df.select_dtypes(include="number").columns
        outlier_cols = 0
        for col in num_cols:
            q1, q3 = source_df[col].quantile(0.25), source_df[col].quantile(0.75)
            iqr = q3 - q1
            if ((source_df[col] < q1 - 1.5*iqr) | (source_df[col] > q3 + 1.5*iqr)).any():
                outlier_cols += 1

        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric("Total records",    f"{source_df.shape[0]:,}")
        qc2.metric("Missing values",   f"{dq_missing}",  f"{dq_missing/dq_total*100:.1f}%" if dq_total else "0%")
        qc3.metric("Duplicate rows",   f"{dq_dupes}")
        qc4.metric("Cols w/ outliers", f"{outlier_cols}")

        if dq_missing == 0 and dq_dupes == 0:
            st.success("✓ Data is clean — no missing values or duplicates detected.")
        else:
            st.warning(f"⚠ {dq_missing} missing values and {dq_dupes} duplicate rows found. "
                       "These will be handled automatically during preprocessing.")

        # ── Auto-clean log ───────────────────────────────────────
        # Auto-detect target column: prefer HVAC_TARGET, fall back to ELEV_TARGET
        if HVAC_TARGET in source_df.columns:
            target_col = HVAC_TARGET
        elif ELEV_TARGET in source_df.columns:
            target_col = ELEV_TARGET
        else:
            target_col = source_df.columns[-1]  # last column as fallback
        cleaned_source, clean_log = automl_clean(source_df.copy(), target_col)
        with st.expander("🧹 Cleaning log", expanded=False):
            for line in clean_log:
                st.markdown(f"- {line}")

        # ── Data preview ─────────────────────────────────────────
        st.markdown("##### Step 3 · Data preview")
        st.dataframe(source_df.head(8), use_container_width=True)
        with st.expander("Full descriptive statistics"):
            st.dataframe(source_df.describe().round(3), use_container_width=True)

        # ── Clear uploaded file + Proceed ────────────────────────
        st.markdown("")
        btn_col1, btn_col2 = st.columns([2, 1])
        with btn_col1:
            if st.button("Proceed to Phase 2 — AI analytics & training ▶", type="primary", key="p1_proceed"):
                st.session_state["workflow_phase"] = 2
                st.rerun()
        with btn_col2:
            if "uploaded_df" in st.session_state:
                if st.button("🗑 Clear uploaded file", key="p1_clear_upload"):
                    del st.session_state["uploaded_df"]
                    st.session_state.pop("uploaded_name", None)
                    st.rerun()

    # ════════════════════════════════════════════════════════════
    # PHASE 2 — AI ANALYTICS & MODEL TRAINING
    # ════════════════════════════════════════════════════════════
    with st.expander("🧠 Phase 2 — AI analytics & model training", expanded=(phase == 2)):
        if phase < 2:
            st.info("Complete Phase 1 first.")
        else:
            # ── Load built-in datasets (cached) ──────────────────
            df      = load_data()
            elev_df = load_elevator_data()

            # ── Resolve active dataset & equipment label ──────────
            # Uploaded file always takes priority over built-in datasets.
            # We detect whether it is an Elevator or HVAC dataset by its columns.
            uploaded_df   = st.session_state.get("uploaded_df", None)
            uploaded_name = st.session_state.get("uploaded_name", "")

            if uploaded_df is not None:
                # Detect dataset type from columns
                if ELEV_TARGET in uploaded_df.columns and HVAC_TARGET not in uploaded_df.columns:
                    active_equipment = "Elevator"
                elif HVAC_TARGET in uploaded_df.columns:
                    active_equipment = equipment  # keep sidebar selection for HVAC variants
                else:
                    active_equipment = equipment
                analyst_df = uploaded_df
                data_label = f"uploaded — `{uploaded_name}`"
            else:
                active_equipment = equipment
                analyst_df = df if equipment != "Elevator" else elev_df
                data_label = f"built-in ({equipment})"

            st.info(f"Training on: {data_label}  |  Equipment context: **{active_equipment}**")

            # ── Train on uploaded data if present, else use cached built-in models ──
            if uploaded_df is not None:
                with st.spinner("Training model on uploaded dataset…"):
                    # Detect target column
                    if HVAC_TARGET in uploaded_df.columns:
                        train_target = HVAC_TARGET
                    elif ELEV_TARGET in uploaded_df.columns:
                        train_target = ELEV_TARGET
                    else:
                        train_target = uploaded_df.columns[-1]

                    cleaned_upload, _ = automl_clean(uploaded_df.copy(), train_target)
                    upload_model, upload_feat_names, upload_X_test, upload_y_test, upload_report =                         automl_train(cleaned_upload, train_target)

                model         = upload_model
                active_report = upload_report
                feat_names    = upload_feat_names
                X_test_ref    = upload_X_test
            else:
                models, reports             = train_models(df)
                elev_model, elev_X_test, elev_report = train_elevator_model(elev_df)
                model         = elev_model if active_equipment == "Elevator" else models[active_equipment]
                active_report = elev_report if active_equipment == "Elevator" else reports[active_equipment]
                feat_names    = get_feature_names(active_equipment)
                X_test_ref    = elev_X_test if active_equipment == "Elevator"                                 else get_training_data(df, active_equipment).sample(50, random_state=1)

            # ── AI Analyst section ───────────────────────────────
            st.markdown("##### Step 4 · AI analyst — fleet health assessment")
            st.caption(
                "The AI analyst examines the cleaned dataset and generates a professional "
                "health report, data quality insights, and feature correlations."
            )
            render_ai_analyst_tab(analyst_df, active_equipment, active_report, model)

            st.markdown("---")

            # ── Model training results ───────────────────────────
            st.markdown("##### Step 5 · Model training results")
            label_key = "1" if "1" in active_report else list(active_report.keys())[0]
            tc1, tc2, tc3, tc4 = st.columns(4)
            tc1.metric("Equipment",  active_equipment)
            tc2.metric("Precision",  f"{active_report[label_key]['precision']:.2%}")
            tc3.metric("Recall",     f"{active_report[label_key]['recall']:.2%}")
            tc4.metric("F1 score",   f"{active_report[label_key]['f1-score']:.2%}")

            # Feature importance chart
            base_clf = model.calibrated_classifiers_[0].estimator
            feat_df  = pd.DataFrame({
                "Feature":    feat_names,
                "Importance": base_clf.feature_importances_,
            }).sort_values("Importance", ascending=False)

            fig_imp, ax_imp = plt.subplots(figsize=(8, 3.5))
            sns.barplot(data=feat_df, x="Importance", y="Feature", ax=ax_imp,
                        palette="flare", orient="h")
            ax_imp.set_title(f"Feature importance — {active_equipment}", fontsize=11)
            fig_imp.tight_layout()
            st.pyplot(fig_imp)
            plt.close(fig_imp)

            # SHAP explainability
            with st.expander("🔍 SHAP explainability (sample)", expanded=False):
                try:
                    X_sample  = X_test_ref.sample(min(50, len(X_test_ref)), random_state=1)
                    explainer = shap.TreeExplainer(base_clf)
                    shap_vals = explainer.shap_values(X_sample)
                    if isinstance(shap_vals, list):
                        sv = shap_vals[1]
                    else:
                        sv = shap_vals[:, :, 1] if shap_vals.ndim == 3 else shap_vals
                    fig_shap, _ = plt.subplots(figsize=(8, 3))
                    shap.summary_plot(sv, X_sample, show=False, plot_size=None)
                    st.pyplot(fig_shap)
                    plt.close(fig_shap)
                except Exception as e:
                    st.warning(f"SHAP unavailable: {e}")

            # ── Proceed button ───────────────────────────────────
            st.markdown("")
            if st.button("Proceed to Phase 3 — prediction & decision ▶", type="primary", key="p2_proceed"):
                st.session_state["workflow_phase"] = 3
                st.rerun()

    # ════════════════════════════════════════════════════════════
    # PHASE 3 — PREDICTION & MAINTENANCE DECISION
    # ════════════════════════════════════════════════════════════
    with st.expander("🔮 Phase 3 — prediction & maintenance decision", expanded=(phase == 3)):
        if phase < 3:
            st.info("Complete Phases 1 and 2 first.")
        else:
            df      = load_data()
            elev_df = load_elevator_data()

            # Resolve model the same way Phase 2 does:
            # uploaded dataset trains a custom model; otherwise use built-in cached models.
            uploaded_df   = st.session_state.get("uploaded_df", None)
            uploaded_name = st.session_state.get("uploaded_name", "")

            if uploaded_df is not None:
                if ELEV_TARGET in uploaded_df.columns and HVAC_TARGET not in uploaded_df.columns:
                    active_equipment = "Elevator"
                else:
                    active_equipment = equipment
                train_target = (HVAC_TARGET if HVAC_TARGET in uploaded_df.columns
                                else ELEV_TARGET if ELEV_TARGET in uploaded_df.columns
                                else uploaded_df.columns[-1])
                cleaned_upload, _ = automl_clean(uploaded_df.copy(), train_target)
                model, p3_feat_names, _, _, _ = automl_train(cleaned_upload, train_target)
            else:
                active_equipment = equipment
                models, reports           = train_models(df)
                elev_model, elev_X_test, _ = train_elevator_model(elev_df)
                model         = elev_model if active_equipment == "Elevator" else models[active_equipment]
                p3_feat_names = get_feature_names(active_equipment)

            # ── Phase 3 sub-tabs ─────────────────────────────────
            p3_tab1, p3_tab2, p3_tab3, p3_tab4 = st.tabs([
                "🔍 Manual prediction",
                "📡 Live simulation",
                "📋 Work orders",
                "🔁 Feedback log",
            ])

            # ── Sensor inputs (inline, not sidebar) ─────────────
            def sensor_inputs(equipment, prefix="p3"):
                st.markdown("##### Enter current sensor readings")
                if equipment == "Elevator":
                    c1, c2 = st.columns(2)
                    feat = {
                        "revolutions": c1.slider("Revolutions (rpm)",  16.0,  94.0,  46.0, 0.1,  key=f"{prefix}_rev"),
                        "humidity":    c2.slider("Humidity (%)",        72.0,  76.0,  74.0, 0.1,  key=f"{prefix}_hum"),
                        "vibration":   c1.slider("Vibration",           2.0,  100.0,  18.0, 0.1,  key=f"{prefix}_vib"),
                        "x1":          c2.slider("Signal x1",          90.0,  168.0, 120.0, 0.1,  key=f"{prefix}_x1"),
                        "x2":          c1.slider("Signal x2",         -57.0,   20.0, -28.0, 0.1,  key=f"{prefix}_x2"),
                        "x3":          c2.slider("Signal x3",           0.23,   1.27,  0.62, 0.01, key=f"{prefix}_x3"),
                        "x4":          c1.slider("Motor load x4",     286.0, 8788.0, 2504.0, 1.0, key=f"{prefix}_x4"),
                        "x5":          c2.slider("Baseline x5",      5241.0, 5686.0, 5510.0, 1.0, key=f"{prefix}_x5"),
                    }
                else:
                    c1, c2 = st.columns(2)
                    if equipment == "Chiller":
                        # Sliders in degC for operator convenience; converted to K for the model
                        st.caption("Temperature inputs are in **degC** and will be converted to Kelvin automatically.")
                        air_c  = c1.slider("Air temperature (degC)",     17.0, 37.0, 25.0, 0.1, key=f"{prefix}_at")
                        proc_c = c2.slider("Process temperature (degC)", 27.0, 47.0, 35.0, 0.1, key=f"{prefix}_pt")
                        feat = {
                            "Air temperature [K]":     round(air_c  + 273.15, 2),
                            "Process temperature [K]": round(proc_c + 273.15, 2),
                            "Rotational speed [rpm]":  c1.slider("Rotational speed (rpm)", 1000, 2000, 1500,        key=f"{prefix}_rs"),
                            "Torque [Nm]":             c2.slider("Torque (Nm)",             30.0,  70.0, 40.0, 0.1, key=f"{prefix}_tq"),
                            "Tool wear [min]":         c1.slider("Tool wear (min)",          0,     300,  50,        key=f"{prefix}_tw"),
                            "Type_L":                  int(c2.checkbox("Type L",                                    key=f"{prefix}_tl")),
                            "Type_M":                  int(c2.checkbox("Type M",                                    key=f"{prefix}_tm")),
                        }
                        c1.caption(f"-> {air_c} degC = {feat['Air temperature [K]']} K")
                        c2.caption(f"-> {proc_c} degC = {feat['Process temperature [K]']} K")
                    else:
                        feat = {
                            "Air temperature [K]":     c1.slider("Air temperature (K)",    290.0, 310.0, 298.0, 0.1, key=f"{prefix}_at"),
                            "Process temperature [K]": c2.slider("Process temperature (K)",300.0, 320.0, 308.0, 0.1, key=f"{prefix}_pt"),
                            "Rotational speed [rpm]":  c1.slider("Rotational speed (rpm)", 1000,  2000,  1500,        key=f"{prefix}_rs"),
                            "Torque [Nm]":             c2.slider("Torque (Nm)",             30.0,  70.0,  40.0, 0.1, key=f"{prefix}_tq"),
                            "Tool wear [min]":         c1.slider("Tool wear (min)",          0,     300,    50,        key=f"{prefix}_tw"),
                            "Type_L":                  int(c2.checkbox("Type L",                                      key=f"{prefix}_tl")),
                            "Type_M":                  int(c2.checkbox("Type M",                                      key=f"{prefix}_tm")),
                        }
                return feat

            # ────────────────────────────────────────────────────
            # P3 TAB 1 — Manual prediction
            # ────────────────────────────────────────────────────
            with p3_tab1:
                st.markdown("##### Step 6 · Run failure prediction")
                st.caption(
                    "Input live or new sensor readings. The trained model scores them "
                    "using a blended signal (65% ML + 35% rule-based) and generates a work order."
                )

                if "last_row_id"        not in st.session_state: st.session_state.last_row_id        = None
                if "last_feedback_given" not in st.session_state: st.session_state.last_feedback_given = False

                features = sensor_inputs(active_equipment, prefix="p3m")

                if st.button("▶ Run prediction", type="primary", key="p3_run"):
                    ml_prob, blended, risk = predict(model, features, active_equipment, p3_feat_names)
                    issues = diagnose(features, active_equipment)
                    action = recommend(risk, issues)
                    row_id = log_prediction(active_equipment, features, blended, risk)
                    wo     = build_work_order(active_equipment, features, blended, risk, issues, action, row_id)
                    st.session_state.last_row_id          = row_id
                    st.session_state.last_feedback_given  = False
                    st.session_state.last_ml_prob         = ml_prob
                    st.session_state.last_blended         = blended
                    st.session_state.last_risk            = risk
                    st.session_state.last_issues          = issues
                    st.session_state.last_action          = action
                    st.session_state.last_wo              = wo

                if st.session_state.last_row_id is not None:
                    ml_prob = st.session_state.last_ml_prob
                    blended = st.session_state.last_blended
                    risk    = st.session_state.last_risk
                    issues  = st.session_state.last_issues
                    action  = st.session_state.last_action
                    wo      = st.session_state.last_wo
                    row_id  = st.session_state.last_row_id

                    st.markdown("---")
                    st.markdown("##### Prediction result")

                    # Outcome banner
                    if risk == "HIGH":
                        st.error(f"⛔ HIGH RISK — {action}")
                    elif risk == "MEDIUM":
                        st.warning(f"⚠️ MEDIUM RISK — {action}")
                    else:
                        st.success(f"✅ LOW RISK — {action}")

                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("Blended risk score", f"{blended:.1%}",
                               help="65% ML model + 35% rule-based issue signal")
                    rc2.metric("Risk level",         risk)
                    rc3.metric("Priority score",     f"${wo['priority_score']:,.0f}",
                               help="Risk probability × estimated repair cost (SGD)")

                    with st.expander("📊 Score breakdown", expanded=True):
                        bc1, bc2 = st.columns(2)
                        bc1.metric("ML model probability", f"{ml_prob:.1%}")
                        n_iss       = len(issues)
                        rule_signal = round(n_iss / len(get_thresholds(active_equipment)), 3)
                        bc2.metric("Rule signal", f"{rule_signal:.1%}",
                                   f"{n_iss} of {len(get_thresholds(active_equipment))} thresholds breached")
                        if ml_prob < 0.30 and n_iss >= 3:
                            st.info(
                                f"ℹ️ ML model alone scores low ({ml_prob:.1%}), but {n_iss} sensors "
                                "are simultaneously in violation. The blended score captures both signals."
                            )

                    st.markdown(f"**Work order:** `{wo['wo_id']}`")

                    # Diagnosed issues
                    if issues:
                        st.markdown("**Diagnosed threshold violations:**")
                        for iss in issues:
                            st.warning(iss)
                    else:
                        st.success("No threshold violations detected.")

                    # Maintenance outcome routing
                    st.markdown("---")
                    st.markdown("##### Step 7 · Maintenance decision")
                    if risk == "HIGH":
                        st.error(
                            "⛔ **Immediate inspection required.**  \n"
                            "Escalate to senior technician. Raise urgent work order. "
                            f"Estimated repair cost: **${REPAIR_COSTS[active_equipment]['HIGH']:,} SGD**."
                        )
                    elif risk == "MEDIUM":
                        st.warning(
                            "⚠️ **Preventive maintenance within 48 h.**  \n"
                            f"Schedule inspection before next operational cycle. "
                            f"Estimated cost if deferred: **${REPAIR_COSTS[active_equipment]['MEDIUM']:,} SGD**."
                        )
                    else:
                        st.success(
                            "✅ **Continue normal monitoring.**  \n"
                            "All signals within acceptable range. Review at next scheduled interval."
                        )

                    # Technician feedback
                    st.markdown("---")
                    st.markdown("##### Step 8 · Technician feedback loop")
                    st.caption(
                        "Your verdict improves the model over time — confirmed failures and false "
                        "alarms are logged to SQLite and used in future retraining."
                    )
                    if st.session_state.last_feedback_given:
                        st.success("✅ Feedback recorded. Run a new prediction to continue.")
                    else:
                        st.caption(f"Recording feedback for **{wo['wo_id']}**")
                        fb1, fb2 = st.columns(2)
                        with fb1:
                            if st.button("✅ Failure confirmed", key="p3_confirm"):
                                record_feedback(row_id, confirmed=True)
                                st.session_state.last_feedback_given = True
                                st.rerun()
                        with fb2:
                            if st.button("❌ False alarm", key="p3_false"):
                                record_feedback(row_id, confirmed=False)
                                st.session_state.last_feedback_given = True
                                st.rerun()

            # ────────────────────────────────────────────────────
            # P3 TAB 2 — Live simulation
            # ────────────────────────────────────────────────────
            with p3_tab2:
                st.markdown("##### Live sensor simulation")
                st.info(
                    "Replays dataset rows as a live sensor feed — simulating real-time BMS data. "
                    "In production, replace `sensor_replay()` with an MQTT subscriber or REST call."
                )

                for k, v in [("sim_running", False), ("sim_history", []),
                              ("sim_step", 0), ("sim_replay_rows", []),
                              ("sim_equipment", active_equipment)]:
                    if k not in st.session_state:
                        st.session_state[k] = v

                n_steps = st.slider("Simulation steps", 5, 50, 20, key="sim_n_steps")
                sb1, sb2 = st.columns(2)
                start = sb1.button("▶ Start simulation", type="primary",
                                   disabled=st.session_state.sim_running, key="sim_start")
                stop  = sb2.button("⏹ Stop", disabled=not st.session_state.sim_running, key="sim_stop")

                if start:
                    src = elev_df if active_equipment == "Elevator" else make_equipment_dataset(df, active_equipment)
                    rows = src[get_feature_names(active_equipment)].sample(
                        n_steps, random_state=int(time.time()) % 9999).to_dict("records")
                    st.session_state.sim_replay_rows = rows
                    st.session_state.sim_history     = []
                    st.session_state.sim_step        = 0
                    st.session_state.sim_running     = True
                    st.session_state.sim_equipment   = active_equipment
                    st.rerun()

                if stop:
                    st.session_state.sim_running = False
                    st.rerun()

                if st.session_state.sim_running:
                    step     = st.session_state.sim_step
                    rows     = st.session_state.sim_replay_rows
                    eq       = st.session_state.sim_equipment
                    eq_model = model  # use the model resolved above (uploaded or built-in)

                    if step < len(rows):
                        live_feat = rows[step]
                        ml_p, bl, rsk = predict(eq_model, live_feat, eq, p3_feat_names)
                        iss = diagnose(live_feat, eq)
                        act = recommend(rsk, iss)
                        st.session_state.sim_history.append({
                            "step": step+1, "prob": bl, "risk": rsk,
                            "action": act, "features": live_feat,
                        })
                        st.session_state.sim_step += 1

                        st.markdown(f"**Step {step+1} / {len(rows)} — {eq}**")
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric("Blended risk score", f"{bl:.1%}")
                        sc2.metric("Risk level", rsk)
                        sc3.metric("Priority score", f"${priority_score(bl, eq, rsk):,.0f}")
                        risk_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(rsk, "⚪")
                        st.markdown(f"{risk_icon} **{act}**")

                        with st.expander("Sensor readings this step"):
                            st.json({k: round(v, 2) for k, v in live_feat.items()})

                        history = st.session_state.sim_history
                        if history:
                            steps_x = [h["step"] for h in history]
                            probs_y = [h["prob"]  for h in history]
                            colors  = ["#E24B4A" if h["risk"]=="HIGH" else
                                       "#EF9F27" if h["risk"]=="MEDIUM" else "#1D9E75"
                                       for h in history]
                            fig, ax = plt.subplots(figsize=(9, 3))
                            ax.plot(steps_x, probs_y, color="#888780", linewidth=1.2, zorder=1)
                            ax.scatter(steps_x, probs_y, c=colors, s=40, zorder=2)
                            ax.axhline(0.55, color="#E24B4A", linestyle="--", linewidth=0.8, label="High (0.55)")
                            ax.axhline(0.28, color="#EF9F27", linestyle="--", linewidth=0.8, label="Medium (0.28)")
                            ax.set_xlabel("Simulation step")
                            ax.set_ylabel("Blended risk score")
                            ax.set_ylim(0, 1)
                            ax.set_title(f"Live failure probability — {eq}", fontsize=11)
                            ax.legend(fontsize=8, loc="upper left")
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                        if len(st.session_state.sim_history) > 1:
                            hist_df = pd.DataFrame([
                                {"Step": h["step"], "Probability": f"{h['prob']:.1%}",
                                 "Risk": h["risk"], "Action": h["action"]}
                                for h in st.session_state.sim_history
                            ])
                            st.dataframe(hist_df, use_container_width=True, hide_index=True)

                        if step + 1 < len(rows):
                            time.sleep(1.2)
                            st.rerun()
                        else:
                            st.session_state.sim_running = False
                            worst = max(st.session_state.sim_history, key=lambda h: h["prob"])
                            sim_row_id = log_prediction(eq, worst["features"], worst["prob"], worst["risk"])
                            st.session_state.sim_worst_row_id = sim_row_id
                            st.rerun()
                    else:
                        st.session_state.sim_running = False
                        st.rerun()

                elif st.session_state.sim_history:
                    history = st.session_state.sim_history
                    steps_x = [h["step"] for h in history]
                    probs_y = [h["prob"]  for h in history]
                    colors  = ["#E24B4A" if h["risk"]=="HIGH" else
                               "#EF9F27" if h["risk"]=="MEDIUM" else "#1D9E75"
                               for h in history]

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

                    n_total  = len(history)
                    n_high   = sum(1 for h in history if h["risk"] == "HIGH")
                    n_medium = sum(1 for h in history if h["risk"] == "MEDIUM")
                    avg_prob = sum(h["prob"] for h in history) / n_total
                    worst    = max(history, key=lambda h: h["prob"])

                    st.markdown("---")
                    st.markdown("##### Simulation conclusion")
                    vc1, vc2, vc3 = st.columns(3)
                    vc1.metric("Avg blended score", f"{avg_prob:.1%}")
                    vc2.metric("High-risk steps",   f"{n_high} / {n_total}")
                    vc3.metric("Peak risk score",   f"{worst['prob']:.1%}",
                               f"Step {worst['step']} — {worst['risk']}")

                    if n_high >= 3 or avg_prob > 0.55:
                        st.error(
                            "⛔ **Maintenance required — schedule immediately.**  \n"
                            f"{n_high} of {n_total} steps flagged HIGH. "
                            f"Peak score {worst['prob']:.1%} at step {worst['step']}."
                        )
                    elif n_high >= 1 or n_medium >= int(n_total * 0.3) or avg_prob > 0.28:
                        st.warning(
                            "⚠️ **Preventive maintenance recommended within 48 h.**  \n"
                            f"{n_high} HIGH + {n_medium} MEDIUM steps detected. "
                            f"Average score {avg_prob:.1%}."
                        )
                    else:
                        st.success(
                            "✅ **Equipment operating normally — no immediate action required.**  \n"
                            f"All {n_total} steps within acceptable range. Average score {avg_prob:.1%}."
                        )

                    with st.expander("Step-by-step breakdown"):
                        hist_df = pd.DataFrame([
                            {"Step": h["step"], "Blended score": f"{h['prob']:.1%}",
                             "Risk": h["risk"], "Action": h["action"]}
                            for h in history
                        ])
                        st.dataframe(hist_df, use_container_width=True, hide_index=True)

            # ────────────────────────────────────────────────────
            # P3 TAB 3 — Work orders
            # ────────────────────────────────────────────────────
            with p3_tab3:
                st.markdown("##### Work order queue")
                st.caption("All predictions ranked by priority score (risk probability × estimated repair cost).")
                fb_df = load_feedback()

                if fb_df.empty:
                    st.info("No predictions logged yet. Run a prediction or simulation first.")
                else:
                    wo_rows = []
                    for _, row in fb_df.iterrows():
                        try:
                            feat = _json.loads(row["features"])
                        except Exception:
                            continue
                        rsk   = row["risk"]
                        eq    = row["equipment"]
                        prob  = row["probability"]
                        score = priority_score(prob, eq, rsk)
                        iss_snap = diagnose(feat, eq)
                        n_iss    = len(iss_snap)
                        if rsk == "HIGH" or n_iss >= 4:        purpose = "⛔ Immediate inspection"
                        elif rsk == "MEDIUM" or n_iss >= 3:    purpose = "⚠️ Preventive maintenance"
                        elif any("aging" in i.lower() for i in iss_snap): purpose = "🛠️ Scheduled overhaul"
                        elif any("temperature" in i.lower() for i in iss_snap): purpose = "🌡️ Cooling system check"
                        elif any("torque" in i.lower() for i in iss_snap): purpose = "🔧 Motor load check"
                        else:                                   purpose = "📋 Routine monitoring"
                        cv = row["confirmed"]
                        status = "✅ Confirmed" if cv == 1 else "❌ False alarm" if cv == 0 else "⏳ Pending"
                        wo_rows.append({
                            "WO ID":        f"WO-{row['id']:05d}",
                            "Timestamp":    row["ts"][:16].replace("T", " "),
                            "Equipment":    eq,
                            "Risk":         rsk,
                            "Score":        f"{prob:.1%}",
                            "Priority ($)": score,
                            "Purpose":      purpose,
                            "Status":       status,
                        })

                    wo_df = pd.DataFrame(wo_rows).sort_values("Priority ($)", ascending=False).reset_index(drop=True)
                    st.dataframe(wo_df, use_container_width=True, hide_index=True)

                    # Retraining opportunity check
                    n_confirmed = int((fb_df["confirmed"] == 1).sum())
                    n_false     = int((fb_df["confirmed"] == 0).sum())
                    st.markdown("---")
                    fc1, fc2 = st.columns(2)
                    fc1.metric("Confirmed failures", n_confirmed)
                    fc2.metric("False alarms",       n_false)
                    if n_confirmed + n_false >= 20:
                        st.success("✓ Sufficient feedback collected — model retrain is eligible.")
                    else:
                        st.info(f"Collect {20 - (n_confirmed + n_false)} more verified verdicts to enable retraining.")

            # ────────────────────────────────────────────────────
            # P3 TAB 4 — Feedback log
            # ────────────────────────────────────────────────────
            with p3_tab4:
                st.markdown("##### Technician feedback log")
                st.caption("All technician verdicts stored in SQLite — used to improve model accuracy over time.")
                fb_df = load_feedback()
                if fb_df.empty:
                    st.info("No feedback logged yet.")
                else:
                    display_cols = ["id", "ts", "equipment", "probability", "risk", "confirmed", "notes"]
                    st.dataframe(
                        fb_df[[c for c in display_cols if c in fb_df.columns]]
                        .rename(columns={"id": "ID", "ts": "Timestamp", "equipment": "Equipment",
                                         "probability": "Score", "risk": "Risk",
                                         "confirmed": "Confirmed", "notes": "Notes"})
                        .sort_values("Timestamp", ascending=False),
                        use_container_width=True, hide_index=True,
                    )

                st.markdown("---")
                if "confirm_clear" not in st.session_state:
                    st.session_state.confirm_clear = False

                if not st.session_state.confirm_clear:
                    if st.button("🗑️ Clear all feedback data", key="p3_clear"):
                        st.session_state.confirm_clear = True
                        st.rerun()
                else:
                    st.warning("This will permanently delete all predictions and feedback. Are you sure?")
                    cy, cn = st.columns(2)
                    with cy:
                        if st.button("Yes, delete everything", type="primary", key="p3_del_yes"):
                            con = sqlite3.connect(DB_PATH)
                            con.execute("DELETE FROM feedback")
                            con.commit()
                            con.close()
                            st.session_state.confirm_clear      = False
                            st.session_state.last_row_id        = None
                            st.session_state.last_feedback_given = False
                            st.success("All data cleared.")
                            st.rerun()
                    with cn:
                        if st.button("Cancel", key="p3_del_no"):
                            st.session_state.confirm_clear = False
                            st.rerun()

    # ── Global reset ─────────────────────────────────────────────
    st.markdown("---")
    if st.button("↩ Reset workflow — start from Phase 1"):
        st.session_state["workflow_phase"] = 1
        st.rerun()



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


# ─────────────────────────────────────────────
# 9. AI ANALYST — helper functions
# ─────────────────────────────────────────────

def _build_analyst_context(df: pd.DataFrame, equipment: str, report: dict) -> str:
    """Summarise dataset stats into a compact context string for the LLM."""
    total   = len(df)
    fail_col = HVAC_TARGET
    failures = int(df[fail_col].sum()) if fail_col in df.columns else "N/A"
    fail_rate = f"{failures/total*100:.2f}%" if isinstance(failures, int) else "N/A"

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    stats_rows = []
    for col in numeric_cols[:12]:          # cap at 12 to stay compact
        s = df[col]
        stats_rows.append(
            f"  {col}: mean={s.mean():.2f}, std={s.std():.2f}, "
            f"min={s.min():.2f}, max={s.max():.2f}"
        )

    label_key = "1" if "1" in report else list(report.keys())[0]
    precision = report[label_key]["precision"]
    recall    = report[label_key]["recall"]
    f1        = report[label_key]["f1-score"]

    ctx = f"""
=== FM PREDICTIVE MAINTENANCE — DATASET CONTEXT ===
Equipment selected : {equipment}
Total records      : {total:,}
Failure events     : {failures}  ({fail_rate} failure rate)
Target column      : {fail_col}
Feature columns    : {', '.join(numeric_cols)}

Numeric feature statistics:
{chr(10).join(stats_rows)}

Trained model performance (test set):
  Precision : {precision:.2%}
  Recall    : {recall:.2%}
  F1 score  : {f1:.2%}

You are an expert FM (Facility Management) data analyst and ML engineer.
Answer questions about this dataset, the model, failure patterns, and
maintenance recommendations using the statistics above.
Keep answers concise, professional, and actionable.
"""
    return ctx


def _stream_ai_response(client, messages: list, context: str) -> str:
    """Call Anthropic with streaming and write chunks to st.write_stream."""
    full_messages = [{"role": "user", "content": context + "\n\n" + messages[0]["content"]}]
    if len(messages) > 1:
        full_messages = (
            [{"role": "user", "content": context + "\n\n" + messages[0]["content"]}]
            + messages[1:]
        )

    collected = []
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=900,
        messages=full_messages,
    ) as stream:
        for text in stream.text_stream:
            collected.append(text)
            yield text
    return "".join(collected)


def _analyst_narrative(client, context: str) -> dict:
    """
    Generate a structured fleet health report as JSON.
    Returns a dict with keys: status, status_color, headline, summary,
    risk_factors (list of {icon, title, detail}),
    recommendations (list of {priority, action, impact}),
    kpis (list of {label, value, delta}).
    """
    import json as _j
    prompt = (
        context
        + """

Return ONLY a JSON object (no markdown fences, no extra text) with this exact structure:
{
  "status": "CRITICAL | WARNING | MODERATE | HEALTHY",
  "headline": "One sharp sentence summarising fleet health (≤20 words)",
  "summary": "2-3 sentence executive summary for an FM manager — plain language, no jargon",
  "kpis": [
    {"label": "Failure Rate", "value": "XX%", "delta": "+X% vs benchmark"},
    {"label": "Model Accuracy", "value": "XX%", "delta": "precision"},
    {"label": "Records Analysed", "value": "XXX,XXX", "delta": "training samples"},
    {"label": "Risk Level", "value": "HIGH | MEDIUM | LOW", "delta": "overall fleet"}
  ],
  "risk_factors": [
    {"icon": "🔴", "title": "Short risk factor title", "detail": "1-2 sentence explanation with specific numbers from the data"},
    {"icon": "🟡", "title": "Second risk factor", "detail": "..."},
    {"icon": "🟠", "title": "Third risk factor", "detail": "..."}
  ],
  "recommendations": [
    {"priority": "P1", "action": "Short action title", "impact": "Expected outcome / benefit", "urgency": "Immediate | Within 48h | Within 7 days"},
    {"priority": "P2", "action": "...", "impact": "...", "urgency": "..."},
    {"priority": "P3", "action": "...", "impact": "...", "urgency": "..."}
  ]
}
Use actual numbers from the dataset context. Be specific and practical. FM manager audience."""
    )
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=900,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    # Strip accidental code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return _j.loads(raw.strip())


def _render_narrative(data: dict):
    """Render the structured fleet health report with rich visual components."""
    status = data.get("status", "MODERATE")
    status_colors = {
        "CRITICAL": ("#E24B4A", "🔴"),
        "WARNING":  ("#EF9F27", "🟡"),
        "MODERATE": ("#378ADD", "🔵"),
        "HEALTHY":  ("#1D9E75", "🟢"),
    }
    color, icon = status_colors.get(status, ("#888780", "⚪"))

    # ── Status banner ────────────────────────────────────────────
    st.markdown(
        f"""<div style="background:{color}18;border-left:5px solid {color};
        border-radius:6px;padding:14px 18px;margin-bottom:16px">
        <span style="font-size:22px">{icon}</span>
        <span style="font-size:18px;font-weight:700;color:{color};margin-left:10px">
        Fleet Status: {status}</span><br>
        <span style="font-size:15px;color:#e0e0e0;margin-left:36px">
        {data.get('headline','')}</span></div>""",
        unsafe_allow_html=True,
    )

    # ── Executive summary ────────────────────────────────────────
    st.markdown(
        f"""<div style="background:#1e2130;border-radius:8px;padding:14px 18px;
        margin-bottom:18px;color:#ccc;font-size:14px;line-height:1.7">
        📋 <strong>Executive Summary</strong><br>{data.get('summary','')}</div>""",
        unsafe_allow_html=True,
    )

    # ── KPI row ──────────────────────────────────────────────────
    kpis = data.get("kpis", [])
    if kpis:
        cols = st.columns(len(kpis))
        for col, kpi in zip(cols, kpis):
            col.metric(kpi.get("label", ""), kpi.get("value", ""), kpi.get("delta", ""))

    st.markdown("---")

    left, right = st.columns(2)

    # ── Risk factors ─────────────────────────────────────────────
    with left:
        st.markdown("#### ⚠️ Key Risk Factors")
        for rf in data.get("risk_factors", []):
            st.markdown(
                f"""<div style="background:#2a2d3e;border-radius:8px;
                padding:12px 15px;margin-bottom:10px">
                <span style="font-size:18px">{rf.get('icon','⚠️')}</span>
                <strong style="font-size:14px;margin-left:8px">{rf.get('title','')}</strong><br>
                <span style="color:#aaa;font-size:13px;margin-left:28px;display:block;margin-top:4px">
                {rf.get('detail','')}</span></div>""",
                unsafe_allow_html=True,
            )

    # ── Recommendations ──────────────────────────────────────────
    with right:
        st.markdown("#### 🛠️ Maintenance Recommendations")
        urgency_colors = {"Immediate": "#E24B4A", "Within 48h": "#EF9F27", "Within 7 days": "#1D9E75"}
        for rec in data.get("recommendations", []):
            urg   = rec.get("urgency", "Within 7 days")
            ucol  = urgency_colors.get(urg, "#888780")
            st.markdown(
                f"""<div style="background:#2a2d3e;border-radius:8px;
                padding:12px 15px;margin-bottom:10px">
                <span style="background:{ucol};color:#fff;font-size:11px;font-weight:700;
                padding:2px 8px;border-radius:4px">{rec.get('priority','')}</span>
                <span style="background:#37415120;color:#aaa;font-size:11px;
                padding:2px 8px;border-radius:4px;margin-left:6px">{urg}</span><br>
                <strong style="font-size:14px;display:block;margin-top:6px">
                {rec.get('action','')}</strong>
                <span style="color:#aaa;font-size:13px">💡 {rec.get('impact','')}</span>
                </div>""",
                unsafe_allow_html=True,
            )


def _analyst_data_quality(df: pd.DataFrame) -> dict:
    """Return a data-quality summary dict."""
    total_cells   = df.shape[0] * df.shape[1]
    missing_total = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    numeric_cols   = df.select_dtypes(include="number").columns.tolist()

    outlier_counts = {}
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        if n_out:
            outlier_counts[col] = n_out

    return {
        "total_cells":    total_cells,
        "missing_total":  missing_total,
        "missing_pct":    round(missing_total / total_cells * 100, 2) if total_cells else 0,
        "duplicate_rows": duplicate_rows,
        "outlier_counts": outlier_counts,
    }


# ─────────────────────────────────────────────
# AI ANALYST TAB — inserted into main()
# ─────────────────────────────────────────────
# (called from within main() below via tab7)

def render_ai_analyst_tab(df, equipment, report, model):
    """Render the entire AI Analyst tab content."""

    st.subheader("🧠 AI Analyst")
    st.caption(
        "Powered by Claude — ask anything about the dataset, model performance, "
        "failure patterns, or maintenance strategy."
    )

    # ── API key input ────────────────────────────────────────────
    st.markdown("#### Connect AI Analyst")
    api_col1, api_col2 = st.columns([3, 1])
    with api_col1:
        api_key = st.text_input(
            "Anthropic API key",
            type="password",
            placeholder="sk-ant-...",
            key="analyst_api_key",
            help="Get your key at console.anthropic.com. It is never stored.",
        )
    with api_col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        connected = bool(api_key)
        if connected:
            st.success("Connected")
        else:
            st.warning("No key")

    if not _ANTHROPIC_AVAILABLE:
        st.error(
            "`anthropic` package not installed. Run: `pip install anthropic`"
        )
        return

    if not api_key:
        st.info(
            "Enter your Anthropic API key above to activate the AI Analyst. "
            "All other tabs work without it."
        )
        return

    try:
        client = _Anthropic(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialise Anthropic client: {e}")
        return

    # Build context once per session (equipment may change)
    ctx_key = f"analyst_ctx_{equipment}"
    if ctx_key not in st.session_state:
        st.session_state[ctx_key] = _build_analyst_context(df, equipment, report)
    context = st.session_state[ctx_key]

    st.markdown("---")

    # ── Section A: Fleet health narrative ───────────────────────
    st.markdown("### Fleet health assessment")
    narr_key = f"analyst_narrative_{equipment}"
    if narr_key not in st.session_state:
        if st.button("Generate fleet health report", key="gen_narrative"):
            with st.spinner("Analysing fleet data..."):
                try:
                    narr = _analyst_narrative(client, context)
                    st.session_state[narr_key] = narr
                    st.rerun()
                except Exception as e:
                    st.error(f"AI error: {e}")
    else:
        _render_narrative(st.session_state[narr_key])
        if st.button("Regenerate", key="regen_narrative"):
            del st.session_state[narr_key]
            st.rerun()

    st.markdown("---")

    # ── Section B: Data quality report ──────────────────────────
    st.markdown("### Data quality report")
    dq = _analyst_data_quality(df)

    dq_c1, dq_c2, dq_c3, dq_c4 = st.columns(4)
    dq_c1.metric("Total cells",    f"{dq['total_cells']:,}")
    dq_c2.metric("Missing values", f"{dq['missing_total']}",
                 f"{dq['missing_pct']}% of data")
    dq_c3.metric("Duplicate rows", dq["duplicate_rows"])
    dq_c4.metric("Columns w/ outliers", len(dq["outlier_counts"]))

    if dq["outlier_counts"]:
        with st.expander("Outlier detail (IQR method)"):
            out_df = pd.DataFrame(
                [{"Column": k, "Outlier rows": v}
                 for k, v in dq["outlier_counts"].items()]
            ).sort_values("Outlier rows", ascending=False)
            st.dataframe(out_df, use_container_width=True, hide_index=True)

    # ── Section C: Feature distributions (top numeric cols) ─────
    st.markdown("### Feature distributions")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    show_cols    = [c for c in numeric_cols if c != HVAC_TARGET][:6]

    if show_cols:
        n_cols = 3
        rows   = [show_cols[i:i+n_cols] for i in range(0, len(show_cols), n_cols)]
        for row_cols in rows:
            plot_cols = st.columns(len(row_cols))
            for ax_col, feat in zip(plot_cols, row_cols):
                with ax_col:
                    fig, ax = plt.subplots(figsize=(3.5, 2.2))
                    if HVAC_TARGET in df.columns:
                        for label, grp in df.groupby(HVAC_TARGET):
                            grp[feat].hist(ax=ax, bins=30, alpha=0.6,
                                           label=f"{'Failure' if label==1 else 'Normal'}",
                                           color="#E24B4A" if label==1 else "#378ADD")
                        ax.legend(fontsize=7)
                    else:
                        df[feat].hist(ax=ax, bins=30, color="#378ADD")
                    ax.set_title(feat, fontsize=9)
                    ax.set_xlabel("")
                    ax.tick_params(labelsize=7)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

    # ── Section D: Correlation heatmap ──────────────────────────
    st.markdown("### Correlation heatmap")
    corr_cols = numeric_cols[:10]
    if len(corr_cols) >= 2:
        corr_matrix = df[corr_cols].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f", ax=ax_corr,
            cmap="coolwarm", linewidths=0.5, linecolor="#2a2d3e",
            annot_kws={"size": 8},
        )
        ax_corr.set_title("Feature correlation matrix", fontsize=11)
        fig_corr.tight_layout()
        st.pyplot(fig_corr)
        plt.close(fig_corr)

    # ── Section E: AI-powered correlations insight ──────────────
    corr_insight_key = f"analyst_corr_{equipment}"
    if len(corr_cols) >= 2:
        if corr_insight_key not in st.session_state:
            if st.button("Explain correlations with AI", key="corr_ai"):
                corr_summary = corr_matrix.unstack().sort_values(ascending=False)
                top_corrs = corr_summary[corr_summary < 1.0].head(6)
                corr_text = "\n".join(
                    [f"  {i[0]} ↔ {i[1]}: {v:.3f}" for i, v in top_corrs.items()]
                )
                with st.spinner("Interpreting correlations..."):
                    try:
                        resp = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=400,
                            messages=[{
                                "role": "user",
                                "content": (
                                    context
                                    + f"\n\nTop feature correlations:\n{corr_text}\n\n"
                                    "In 3-4 sentences, explain what these correlations "
                                    "mean for equipment failure in FM terms. "
                                    "Be specific and practical."
                                ),
                            }],
                        )
                        st.session_state[corr_insight_key] = resp.content[0].text
                        st.rerun()
                    except Exception as e:
                        st.error(f"AI error: {e}")
        else:
            st.info(st.session_state[corr_insight_key])
            if st.button("Clear", key="clear_corr"):
                del st.session_state[corr_insight_key]
                st.rerun()

    st.markdown("---")

    # ── Section F: Natural language Q&A chat ────────────────────
    st.markdown("### Ask the AI Analyst")
    st.caption(
        "Ask anything: failure patterns, feature importance, maintenance schedules, "
        "what-if scenarios, or model behaviour."
    )

    # Suggested questions
    suggestions = [
        "What are the main causes of failure for this equipment?",
        "Which features are most predictive of failure?",
        "What tool wear threshold should trigger maintenance?",
        "How does temperature affect failure probability?",
        "Suggest a preventive maintenance schedule based on this data.",
    ]
    st.markdown("**Suggested questions:**")
    sug_cols = st.columns(len(suggestions))
    for i, (col, q) in enumerate(zip(sug_cols, suggestions)):
        with col:
            if st.button(q[:38] + "…" if len(q) > 38 else q, key=f"sug_{i}",
                         use_container_width=True):
                st.session_state.analyst_pending_q = q

    # Chat history
    if "analyst_chat" not in st.session_state:
        st.session_state.analyst_chat = []   # list of {"role": ..., "content": ...}

    # Display existing conversation
    for msg in st.session_state.analyst_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle suggested question injection
    pending = st.session_state.pop("analyst_pending_q", None)

    # Chat input
    user_input = st.chat_input("Ask the AI Analyst…") or pending

    if user_input:
        # Append user message
        st.session_state.analyst_chat.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build message list (context injected into first user turn)
        history = st.session_state.analyst_chat
        if len(history) == 1:
            api_messages = [{"role": "user", "content": context + "\n\n" + user_input}]
        else:
            api_messages = (
                [{"role": history[0]["role"],
                  "content": context + "\n\n" + history[0]["content"]}]
                + [{"role": m["role"], "content": m["content"]} for m in history[1:]]
            )

        # Stream response
        with st.chat_message("assistant"):
            try:
                collected = []
                response_placeholder = st.empty()
                with client.messages.stream(
                    model="claude-sonnet-4-20250514",
                    max_tokens=700,
                    messages=api_messages,
                ) as stream:
                    for chunk in stream.text_stream:
                        collected.append(chunk)
                        response_placeholder.markdown("".join(collected) + "▌")
                full_response = "".join(collected)
                response_placeholder.markdown(full_response)
                st.session_state.analyst_chat.append(
                    {"role": "assistant", "content": full_response}
                )
            except Exception as e:
                st.error(f"AI error: {e}")

    # Clear chat button
    if st.session_state.analyst_chat:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.analyst_chat = []
            st.rerun()


if __name__ == "__main__":
    main()
