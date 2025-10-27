import json
from pathlib import Path
from typing import Dict, Any, List

import altair as alt
import pandas as pd
import streamlit as st

# ----------------------------
# Config
# ----------------------------
APP_TITLE = "Acme Retail: Architecture Trade-off Simulation"
BASELINE = {"time_weeks": 12, "cost_k": 200, "quality": 70}
MIN_MAX = {
    "time_weeks": (2, 30),   # for chart scaling
    "cost_k": (50, 400),
    "quality": (0, 100)
}

# ----------------------------
# Utilities
# ----------------------------
@st.cache_data
def load_questions(path: str = "questions.json") -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)

def apply_deltas(selections: Dict[str, str], questions: List[Dict[str, Any]]) -> Dict[str, float]:
    result = BASELINE.copy()
    for q in questions:
        sel = selections.get(q["id"])
        if not sel:
            continue
        opt = next((o for o in q["options"] if o["key"] == sel), None)
        if opt:
            for k, v in opt["deltas"].items():
                result[k] = result.get(k, 0) + v
    # clamp quality to [0,100]
    result["quality"] = max(0, min(100, result["quality"]))
    return result

def progress(current_index: int, total: int) -> float:
    return (current_index + 1) / total

def outcome_dataframe(outcome: Dict[str, float]) -> pd.DataFrame:
    # friendlier labels
    data = [
        {"Metric": "Time (weeks)", "Value": outcome["time_weeks"]},
        {"Metric": "Cost (k USD)", "Value": outcome["cost_k"]},
        {"Metric": "Quality (0-100)", "Value": outcome["quality"]},
    ]
    return pd.DataFrame(data)

def outcome_chart(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Value:Q"),
            y=alt.Y("Metric:N", sort=None),
            tooltip=["Metric", "Value"]
        )
        .properties(height=180)
    )

# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="centered")
st.title(APP_TITLE)
st.caption("Choose your options. Your decisions impact Time, Cost, and Quality.")

questions = load_questions()
total_qs = len(questions)

if "idx" not in st.session_state:
    st.session_state.idx = 0
if "selections" not in st.session_state:
    st.session_state.selections = {}

# Sidebar: current outcome (live)
with st.sidebar:
    st.header("Live Outcome")
    outcome_live = apply_deltas(st.session_state.selections, questions)
    st.metric("Time (weeks)", outcome_live["time_weeks"])
    st.metric("Cost (k USD)", outcome_live["cost_k"])
    st.metric("Quality", outcome_live["quality"])
    st.divider()
    if st.button("Reset choices", type="secondary"):
        st.session_state.selections = {}
        st.session_state.idx = 0
        st.rerun()

# Progress
st.progress(progress(st.session_state.idx, total_qs))

# Current question
q = questions[st.session_state.idx]
st.subheader(f"{st.session_state.idx + 1}/{total_qs} — {q['label']}")
st.write(q["prompt"])

# Options
keys = [opt["key"] for opt in q["options"]]
labels = [opt["label"] for opt in q["options"]]

# map stored selection to index
default_index = 0
if q["id"] in st.session_state.selections:
    try:
        default_index = keys.index(st.session_state.selections[q["id"]])
    except ValueError:
        default_index = 0

choice_idx = st.radio(
    "Select one:",
    options=list(range(len(keys))),
    format_func=lambda i: labels[i],
    index=default_index if q["id"] in st.session_state.selections else 0,
    key=f"radio_{q['id']}"
)

# Save selection
st.session_state.selections[q["id"]] = keys[choice_idx]

# Navigation buttons
cols = st.columns([1,1,2])
with cols[0]:
    if st.button("← Back", disabled=st.session_state.idx == 0):
        st.session_state.idx -= 1
        st.rerun()
with cols[1]:
    if st.button("Next →", disabled=st.session_state.idx >= total_qs - 1):
        st.session_state.idx += 1
        st.rerun()

# If last question, show results & download
if st.session_state.idx == total_qs - 1:
    st.divider()
    st.subheader("Your Outcome")
    final_outcome = apply_deltas(st.session_state.selections, questions)
    df = outcome_dataframe(final_outcome)
    st.altair_chart(outcome_chart(df), use_container_width=True)

    # Friendly explanation
    st.markdown(
        f"""
**Summary**
- **Estimated time to launch:** `{final_outcome['time_weeks']:.0f}` weeks  
- **Estimated cost:** `${final_outcome['cost_k']:.0f}k`  
- **Quality score:** `{final_outcome['quality']:.0f}` / 100

> Lower **Time** & **Cost** are better; higher **Quality** is better. We can discuss the trade-offs behind each choice and how to mitigate downsides in the architecture.
"""
    )

    # Download selections & outcome
    export = {
        "selections": st.session_state.selections,
        "outcome": final_outcome
    }
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

# Footer / facilitation help
st.divider()
with st.expander("Facilitation Notes (hidden during demo)"):
    st.write("""
- Keep the group moving; aim to complete choices in ~10–12 minutes.
- Then discuss trade-offs and mitigation (e.g., start with batch CSV but add dbt tests & observability).
- Close with a proposed architecture and phased rollout plan.
""")
