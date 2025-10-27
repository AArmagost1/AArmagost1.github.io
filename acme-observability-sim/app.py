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

# Targets (also serve as baselines for Time/Cost; Quality baseline is 0)
TARGETS = {
    "time_weeks": 12.0,   # pass if <=
    "cost_k": 250.0,      # pass if <=
    "quality": 84.0       # pass if >=
}

# Baselines used to accumulate deltas for the running outcome
BASELINE = {
    "time_weeks": TARGETS["time_weeks"],  # 12
    "cost_k": TARGETS["cost_k"],          # 250
    "quality": 0.0                        # start at 0, build up toward 84
}

# Weights for final composite score (must sum to 1.0)
WEIGHTS = {
    "time_weeks": 0.30,
    "cost_k": 0.30,
    "quality": 0.40
}

# For charts scaling (cosmetic)
MIN_MAX = {
    "time_weeks": (2, 30),
    "cost_k": (50, 400),
    "quality": (0, 100)
}

# =========================
# Default questions (used if questions.json not found)
# =========================
DEFAULT_QUESTIONS = [
    {
        "id": "ingest",
        "label": "Data Source Strategy",
        "prompt": "How will you ingest partner data into the platform?",
        "options": [
            {"key": "batch_csv", "label": "Batch nightly CSV drops",
             "deltas": {"time_weeks": -2, "cost_k": -40, "quality": -8}},
            {"key": "partner_apis", "label": "Direct partner API integrations (daily sync)",
             "deltas": {"time_weeks": 1, "cost_k": 20, "quality": 8}},
            {"key": "streaming", "label": "Real-time streaming (Kafka/Event Hub)",
             "deltas": {"time_weeks": 3, "cost_k": 50, "quality": 14}}
        ]
    },
    {
        "id": "warehouse",
        "
