# app.py
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st

# =========================
# App Config
# =========================
APP_TITLE = "Acme Retail: Architecture Trade-off Simulation"

# Baselines (Time & Cost accumulate from here; Quality is a running score starting at 0)
BASELINE = {"time_weeks": 12.0, "cost_k": 200.0, "quality": 0.0}

# Targets for pass/fail & scoring
TARGETS = {
    "time_weeks": 12.0,   # pass if <=
    "cost_k": 250.0,      # pass if <=
    "quality": 84.0       # pass if >=
}

# Weights for composite score (sum to 1.0)
WEIGHTS = {
    "time_weeks": 0.30,
    "cost_k": 0.30,
    "quality": 0.40
}

# Chart scaling (cosmetic)
MIN_MAX = {
    "time_weeks": (2, 30),
    "cost_k": (50, 400),
    "quality": (0, 100)
}

# =========================
# Default questions (used if questions.json not found)
# Q1 deltas align with your examples:
# - Batch nightly CSV → Time 10 (12-2), Cost 210 (200+10), Quality 8 (0+8)
# - Streaming → Time 15 (12+3), Cost 300 (200+100), Quality 14 (0+14)
# =========================
DEFAULT_QUESTIONS = [
    {
        "id": "ingest",
        "label": "Data Source Strategy",
        "prompt": "How will you ingest partner data into the platform?",
        "options": [
            {"key": "batch_csv", "label": "Batch nightly CSV drops",
             "deltas": {"time_weeks": -2, "cost_k": 10, "quality": 8}},
            {"key": "partner_apis", "label": "Direct partner API integrations (daily sync)",
             "deltas": {"time_weeks": 1, "cost_k": 20, "quality": 10}},
            {"key": "streaming", "label": "Real-time streaming (Kafka/Event Hub)",
             "deltas": {"time_weeks": 3, "cost_k": 100, "quality": 14}}
        ]
    },
    {
        "id": "warehouse",
        "label": "Analytics Warehouse",
        "prompt": "Where will you store & analyze data?",
        "options": [
            {"key": "on_prem", "label": "Existing on-prem database",
             "deltas": {"time_weeks": 4, "cost_k": 10, "quality": -6}},
            {"key": "snowflake", "label": "Cloud DW (Snowflake)",
             "deltas": {"time_weeks": -1, "cost_k": 5, "quality": 12}}
        ]
    },
    {
        "id": "delivery",
        "label": "Analytics Delivery",
        "prompt": "How will customers consume analytics?",
        "options": [
            {"key": "internal_only", "label": "Internal dashboards only",
             "deltas": {"time_weeks": -2, "cost_k": -15, "quality": -12}},
            {"key": "embedded", "label": "Embedded customer-facing dashboards",
             "deltas": {"time_weeks": 1, "cost_k": 20, "quality": 10}}
        ]
    },
    {
        "id": "governance",
        "label": "Governance & Security",
        "prompt": "Pick your initial governance approach.",
        "options": [
            {"key": "minimal", "label": "Minimal controls to move fast",
             "deltas": {"time_weeks": -1, "cost_k": -10, "quality": -10}},
            {"key": "central_rbac", "label": "Centralized RBAC + masking policies",
             "deltas": {"time_weeks": 1, "cost_k": 10, "quality": 12}}
        ]
    },
    {
        "id": "modeling",
        "label": "Transformations & Modeling",
        "prompt": "How will you transform and model data?",
        "options": [
            {"key": "sql_sprawl", "label": "Ad-hoc SQL scripts per team",
             "deltas": {"time_weeks": -1, "cost_k": -5, "quality": -8}},
            {"key": "dbt", "label": "dbt with tests, CI, and docs",
             "deltas": {"time_weeks": 1, "cost_k": 8, "quality": 14}}
        ]
    },
    {
        "id": "sla",
        "label": "SLAs & Observability",
        "prompt": "What level of observability will you start with?",
        "options": [
            {"key": "basic", "label": "Basic pipeline alerts",
             "deltas": {"time_weeks": 0, "cost_k": -5, "quality": 4}},
            {"key": "full", "label": "End-to-end observability + cost monitoring",
             "deltas": {"time_weeks": 1, "cost_k": 12, "quality": 12}}
        ]
    }
]

# =========================
# Load Questions
# =========================
@st.cache_data
def load_questions(path: str | None = None) -> List[Dict[str, Any]]:
    """
    Load questions from questions.json next to app.py; if missing, use defaults.
    """
    try:
        base_dir = Path(__file__).parent
        q_path = Path(path) if path else (base_dir / "questions.json")
        with q_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load questions.json ({e}). Using built-in defaults.")
        return DEFAULT_QUESTIONS

# =========================
# Core Logic
# =========================
def apply_deltas(selections: Dict[str, str], questions: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Apply selected option deltas cumulatively to the baseline.
    Quality is a running score starting at 0.
    Only answered questions contribute.
    """
    result = BASELINE.copy()
    for q in questions:
        sel = selections.get(q["id"])
        if not sel:
            continue
        opt = next((o for o in q["options"] if o["key"] == sel), None)
        if opt:
            for k, v in opt["deltas"].items():
                result[k] = result.get(k, 0.0) + float(v)
    # Clamp quality [0,100]
    result["quality"] = max(0.0, min(100.0, result["quality"]))
    return result

def outcome_dataframe(outcome: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Metric": "Time (weeks)", "Key": "time_weeks", "Value": outcome["time_weeks"], "Target": TARGETS["time_weeks"], "Direction": "≤"},
            {"Metric": "Cost (k USD)", "Key": "cost_k", "Value": outcome["cost_k"], "Target": TARGETS["cost_k"], "Direction": "≤"},
            {"Metric": "Quality (0-100)", "Key": "quality", "Value": outcome["quality"], "Target": TARGETS["quality"], "Direction": "≥"},
        ]
    )

def metric_pass_fail(key: str, value: float) -> Tuple[bool, str]:
    tgt = TARGETS[key]
    if key in ("time_weeks", "cost_k"):
        ok = value <= tgt
        lbl = "Pass" if ok else "Miss"
        return ok, lbl
    elif key == "quality":
        ok = value >= tgt
        lbl = "Pass" if ok else "Miss"
        return ok, lbl
    return False, "N/A"

def per_metric_score(key: str, value: float) -> float:
    """
    0-100 score vs target.
    - For time/cost: lower is better → target/value
    - For quality: higher is better → value/target
    """
    tgt = TARGETS[key]
    if key in ("time_weeks", "cost_k"):
        if value <= 0:
            return 100.0
        return float(min(100.0, max(0.0, 100.0 * tgt / value)))
    elif key == "quality":
        if tgt <= 0:
            return 100.0
        return float(min(100.0, max(0.0, 100.0 * value / tgt)))
    return 0.0

def composite_score(outcome: Dict[str, float]) -> float:
    time_s = per_metric_score("time_weeks", outcome["time_weeks"])
    cost_s = per_metric_score("cost_k", outcome["cost_k"])
    qual_s = per_metric_score("quality", outcome["quality"])
    score = (
        WEIGHTS["time_weeks"] * time_s +
        WEIGHTS["cost_k"] * cost_s +
        WEIGHTS["quality"] * qual_s
    )
    return round(score, 1)

def outcome_chart(df: pd.DataFrame, show_targets: bool = True) -> alt.Chart:
    base = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Value:Q", title="Value"),
            y=alt.Y("Metric:N", sort=None),
            tooltip=["Metric:N", "Value:Q", "Target:Q"]
        )
        .properties(height=180)
    )
    if show_targets:
        rules = alt.Chart(df).mark_rule(strokeDash=[4, 4]).encode(
            x="Target:Q",
            y="Metric:N",
            tooltip=["Metric:N", "Target:Q"]
        )
        return base + rules
    return base

# =========================
# UI Helpers
# =========================
def show_target_badges(df: pd.DataFrame):
    cols = st.columns(3)
    for i, (_, row) in enumerate(df.iterrows()):
        key = row["Key"]
        val = row["Value"]
        tgt = row["Target"]
        ok, _ = metric_pass_fail(key, val)
        with cols[i]:
            st.markdown(
                f"**{row['Metric']}**: `{val:.1f} / {tgt:.1f}` — "
                + (":green[Pass]" if ok else ":red[Miss]")
            )

def show_composite_score(outcome: Dict[str, float]):
    score = composite_score(outcome)
    t_s = per_metric_score("time_weeks", outcome["time_weeks"])
    c_s = per_metric_score("cost_k", outcome["cost_k"])
    q_s = per_metric_score("quality", outcome["quality"])

    st.subheader(f"Composite Score: **{score}/100**")
    st.caption(
        f"Weighted vs target — Time {WEIGHTS['time_weeks']*100:.0f}%, Cost {WEIGHTS['cost_k']*100:.0f}%, Quality {WEIGHTS['quality']*100:.0f}%"
    )
    cols = st.columns(3)
    cols[0].metric("Time score", f"{t_s:.0f}")
    cols[1].metric("Cost score", f"{c_s:.0f}")
    cols[2].metric("Quality score", f"{q_s:.0f}")

# =========================
# App
# =========================
st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="centered")
st.title(APP_TITLE)
st.caption("Choose your options. Your decisions impact Time, Cost, and Quality.")

questions = load_questions()
total_qs = len(questions)

# Init session state
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "selections" not in st.session_state:
    st.session_state.selections = {}
if "show_results" not in st.session_state:
    st.session_state.show_results = False

# Progress
st.progress((st.session_state.idx + 1) / total_qs)

# Current question content FIRST (so state updates before sidebar is computed)
q = questions[st.session_state.idx]
st.subheader(f"{st.session_state.idx + 1}/{total_qs} — {q['label']}")
st.write(q["prompt"])

# Build options with placeholder — no default selection
opt_pairs = [(None, "— Select an option —")] + [(opt["key"], opt["label"]) for opt in q["options"]]

# Resolve current selection (if any)
current_key = st.session_state.selections.get(q["id"], None)

def index_for_key(k, pairs):
    for i, (key, _) in enumerate(pairs):
        if key == k:
            return i
    return 0  # placeholder

# Use index=None behavior if Streamlit supports it; otherwise compute index
try:
    selected_idx = st.selectbox(
        "Select one:",
        options=list(range(len(opt_pairs))),
        format_func=lambda i: opt_pairs[i][1],
        index=(None if current_key is None else index_for_key(current_key, opt_pairs)),
        key=f"select_{q['id']}"
    )
except TypeError:
    # Fallback for older Streamlit (index cannot be None)
    selected_idx = st.selectbox(
        "Select one:",
        options=list(range(len(opt_pairs))),
        format_func=lambda i: opt_pairs[i][1],
        index=index_for_key(current_key, opt_pairs),
        key=f"select_{q['id']}"
    )

selected_key = opt_pairs[selected_idx][0] if selected_idx is not None else None

# Persist only real selections (no placeholder)
if selected_key is None:
    if q["id"] in st.session_state.selections:
        del st.session_state.selections[q["id"]]
else:
    st.session_state.selections[q["id"]] = selected_key

# Navigation buttons
col_back, col_next, col_spacer = st.columns([1, 1, 2])
with col_back:
    if st.button("← Back", disabled=(st.session_state.idx == 0)):
        st.session_state.idx -= 1
        st.rerun()
with col_next:
    # Optional: require selection before proceeding
    disable_next = (st.session_state.idx < total_qs - 1) and (q["id"] not in st.session_state.selections)
    if st.button("Next →", disabled=disable_next):
        st.session_state.idx += 1
        st.rerun()

# Finish gate on last question
if st.session_state.idx == total_qs - 1 and not st.session_state.show_results:
    st.divider()
    # Optional: require final selection before Finish
    can_finish = (q["id"] in st.session_state.selections)
    if st.button("Finish & Show Outcome", type="primary", disabled=not can_finish):
        st.session_state.show_results = True
        st.rerun()

# === Sidebar AFTER selection is stored so it reflects the latest state ===
with st.sidebar:
    st.header("Live Outcome")
    live = apply_deltas(st.session_state.selections, questions)
    st.metric("Time (weeks)", f"{live['time_weeks']:.1f} / {TARGETS['time_weeks']:.1f}")
    st.metric("Cost (k USD)", f"{live['cost_k']:.1f} / {TARGETS['cost_k']:.1f}")
    st.metric("Quality", f"{live['quality']:.1f} / {TARGETS['quality']:.1f}")
    st.divider()
    if st.button("Reset choices", type="secondary"):
        st.session_state.selections = {}
        st.session_state.idx = 0
        st.session_state.show_results = False
        st.rerun()

# Results (after Finish) — updates dynamically as selections change
if st.session_state.show_results:
    st.divider()
    st.subheader("Your Outcome")

    final_outcome = apply_deltas(st.session_state.selections, questions)
    df = outcome_dataframe(final_outcome)

    st.altair_chart(outcome_chart(df, show_targets=True), use_container_width=True)
    show_target_badges(df)
    show_composite_score(final_outcome)

    st.markdown(
        f"""
**Summary**
- **Estimated time to launch:** `{final_outcome['time_weeks']:.1f} / {TARGETS['time_weeks']:.1f}` weeks
- **Estimated cost:** `${final_outcome['cost_k']:.1f}k / {TARGETS['cost_k']:.1f}k`
- **Quality score:** `{final_outcome['quality']:.1f} / {TARGETS['quality']:.1f}`

> Lower **Time** & **Cost** are better; higher **Quality** is better. The composite score reflects weighted performance vs targets.
"""
    )

    # Export
    export_df = pd.DataFrame(
        [{"type": "selection", "key": k, "value": v} for k, v in st.session_state.selections.items()] +
        [{"type": "outcome", "key": k, "value": v} for k, v in final_outcome.items()]
    )
    st.download_button(
        "Download decisions (.csv)",
        data=export_df.to_csv(index=False),
        file_name="acme_decisions.csv",
        mime="text/csv"
    )

    st.info("You can go back and change choices — the outcome and score will update automatically.")

# Facilitation notes
st.divider()
with st.expander("Facilitation Notes (hidden during demo)"):
    st.write("""
- No default selections — Live Outcome remains at baseline until each choice is made.
- Sidebar updates immediately because it's rendered after the selection logic.
- Encourage toggling options to see trade-offs in real time.
- After 'Finish', discuss mitigations and a phased architecture to move toward targets.
""")
