# ai_agent.py
"""
Gemini 2.0 AI agent for the chiller dashboard.

- Uses google-generativeai (Gemini 2.0 Flash Experimental).
- Single entry point: ai_answer(query, df, chiller)
  which is already used by app.py on all 4 pages.
"""

import os
import pandas as pd
import streamlit as st
import google.generativeai as genai


# ---------------------------------------------------------------------
# Configure Gemini 2.0
# ---------------------------------------------------------------------
# Prefer Streamlit secrets if available, else environment variable
API_KEY = None
try:
    API_KEY = st.secrets.get("GOOGLE_API_KEY", None)
except Exception:
    # st.secrets may not be available in some contexts (e.g. offline test)
    API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    # Don't crash the app – just warn and return a friendly message later
    st.warning(
        "⚠ GOOGLE_API_KEY is not configured. "
        "Add it in Streamlit → Settings → Secrets as GOOGLE_API_KEY."
    )

# Gemini 2.0 Flash experimental text model
MODEL_NAME = "gemini-2.0-flash-exp"


# ---------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------
def _build_prompt(query: str, df: pd.DataFrame, chiller: str) -> str:
    """
    Build a compact but rich prompt using latest telemetry from the selected chiller.
    Works for all 4 pages:
      - Power Consumption
      - Anomaly Detection
      - Predictive Maintenance
      - Design Power / KPIs
    """

    # Take a small slice of the most recent data for that chiller
    recent = df[df["chiller"] == chiller].sort_values("time").tail(24)

    # Only keep columns that are most relevant (if they exist)
    keep_cols = [
        col
        for col in recent.columns
        if col
        in [
            "time",
            "ambient",
            "it_load",
            "chw_in",
            "power_predicted",
            "power_actual",
            "anomaly_score",
        ]
    ]
    recent_view = recent[keep_cols] if keep_cols else recent

    return f"""
You are an expert Data Center & BMS AI assistant focusing on chiller analytics.

Context:
- The user is viewing a dashboard for chiller performance, anomaly detection,
  predictive maintenance, and design-power / health KPIs.
- Each question below comes from one of these pages and is about deviations in graphs,
  thresholds, priorities, alerts, or operating conditions.

User question:
\"\"\"{query}\"\"\"  

Selected chiller: {chiller}

Here is the latest telemetry for this chiller (most recent rows at the bottom):

{recent_view.to_string(index=False)}

Columns (if present) mean:
- time: timestamp of the sample
- ambient: ambient (outdoor) air temperature in °C
- it_load: IT load (% of design)
- chw_in: chilled water inlet temperature in °C
- power_predicted: model-predicted power draw (design / expected)
- power_actual: actual measured power draw
- anomaly_score: positive = unusual/high deviation, negative = unusually low

TASK:
1. First, directly answer the user's question in the context of this data.
2. Explain WHY the graphs might show deviations or changes:
   - e.g., higher ambient temperature, spike in IT load, higher CHW inlet temp,
     degraded performance, sensor issues, etc.
3. Explicitly name which parameter(s) are driving the deviation
   (ambient, it_load, chw_in, anomaly_score, power_actual vs power_predicted).
4. Explain impact for the operations team:
   - energy/cost impact,
   - risk to cooling capacity or uptime,
   - maintenance or performance implications.
5. Give 3–6 concrete, practical actions the operations team should consider:
   - setpoint changes,
   - load redistribution,
   - maintenance checks,
   - threshold tuning or alert configuration.

Format the answer clearly with short paragraphs and bullet points.
Avoid overly long essays. Keep it crisp and operational.
"""


# ---------------------------------------------------------------------
# Public function used by app.py
# ---------------------------------------------------------------------
def ai_answer(query: str, df: pd.DataFrame, chiller: str) -> str:
    """
    Main entry point for the dashboard.

    Parameters
    ----------
    query : str
        The natural language question selected in the dropdown
        (different per page: power, anomaly, maintenance, KPIs).
    df : pd.DataFrame
        Full time-series dataframe for all chillers (simulated).
    chiller : str
        Chiller ID selected in the UI.

    Returns
    -------
    str : AI-generated explanation / root cause / recommendation text.
    """
    if not API_KEY:
        return (
            "⚠ Gemini 2.0 AI is not configured yet. "
            "Please set GOOGLE_API_KEY in Streamlit Secrets."
        )

    prompt = _build_prompt(query, df, chiller)

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)

        # Different SDK versions expose text slightly differently
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        # Fallback: stringify entire response
        return str(response)

    except Exception as e:
        return f"❌ Gemini 2.0 error: {e}"
