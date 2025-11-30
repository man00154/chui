# ai_agent.py
"""
Gemini + offline fallback AI agent for the chiller dashboard.

- Tries Google Gemini (gemini-1.5-flash-8b-latest).
- If key is missing or quota/429/other errors occur,
  falls back to a local rule-based explanation (no API, free).
"""

import os
import pandas as pd
import streamlit as st

# Try to import Gemini SDK, but don't crash if it's missing
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ---------------------------------------------------------------------
# Config: API key + model
# ---------------------------------------------------------------------
API_KEY = None
try:
    API_KEY = st.secrets.get("GOOGLE_API_KEY", None)
except Exception:
    API_KEY = os.getenv("GOOGLE_API_KEY")

GEMINI_MODEL_NAME = "gemini-1.5-flash-8b-latest"  # current lightweight model (2025)


if GEMINI_AVAILABLE and API_KEY:
    genai.configure(api_key=API_KEY)
elif GEMINI_AVAILABLE and not API_KEY:
    st.info(
        "ℹ GOOGLE_API_KEY not configured, will use offline rule-based explanation instead."
    )
else:
    st.info(
        "ℹ google-generativeai not installed, will use offline rule-based explanation."
    )


# ---------------------------------------------------------------------
# Offline fallback: rule-based explanation (no external API)
# ---------------------------------------------------------------------
def _local_rule_based_answer(query: str, df: pd.DataFrame, chiller: str) -> str:
    """
    Generate a deterministic, 'AI-style' explanation using only local data.

    This costs nothing, uses no external model, and works even with no API key
    or when remote quotas are exhausted.
    """
    df_sel = df[df["chiller"] == chiller]

    if df_sel.empty:
        return (
            f"For chiller {chiller}, no data is available in the current "
            "simulation window. Please verify the configuration."
        )

    # Basic stats
    ambient_mean = df_sel["ambient"].mean() if "ambient" in df_sel else None
    it_mean = df_sel["it_load"].mean() if "it_load" in df_sel else None
    chw_mean = df_sel["chw_in"].mean() if "chw_in" in df_sel else None
    anomaly_mean = df_sel["anomaly_score"].mean() if "anomaly_score" in df_sel else None

    power_diff = None
    if "power_actual" in df_sel and "power_predicted" in df_sel:
        power_diff = (df_sel["power_actual"] - df_sel["power_predicted"]).mean()

    bullets = []

    # Interpret the query category a bit
    q_lower = query.lower()

    if "ambient" in q_lower:
        bullets.append(
            "• Higher ambient temperature generally forces chillers to work harder, "
            "increasing compressor power and shifting the power curve upward."
        )
    if "it load" in q_lower:
        bullets.append(
            "• An increase in IT load raises the heat rejection requirement, which in turn "
            "drives higher chilled-water demand and chiller power."
        )
    if "chilled water inlet" in q_lower or "chw" in q_lower:
        bullets.append(
            "• Higher chilled water inlet temperature often indicates reduced heat transfer "
            "efficiency across coils, which can lead to higher power for the same load."
        )
    if "anomaly" in q_lower:
        bullets.append(
            "• High anomaly scores occur when the difference between predicted and actual power "
            "or other variables exceeds the learned normal band."
        )
    if "maintenance" in q_lower or "priority" in q_lower:
        bullets.append(
            "• Maintenance priority is usually driven by a combination of anomaly frequency, "
            "magnitude of deviation, age/runtime of equipment, and business criticality."
        )
    if "kpi" in q_lower or "threshold" in q_lower or "alert" in q_lower:
        bullets.append(
            "• Health KPIs and thresholds should be tuned so that they catch early degradation "
            "without generating excessive nuisance alarms."
        )

    # Simple data-driven comments
    if ambient_mean is not None:
        bullets.append(
            f"• In the current simulation, mean ambient temperature for {chiller} is "
            f"around {ambient_mean:.1f} °C."
        )
    if it_mean is not None:
        bullets.append(
            f"• The average IT load is ~{it_mean:.1f}% of design, which influences both "
            f"cooling demand and power consumption."
        )
    if chw_mean is not None:
        bullets.append(
            f"• The average chilled water inlet temperature is ~{chw_mean:.1f} °C in this period."
        )
    if anomaly_mean is not None:
        bullets.append(
            f"• The mean anomaly score for {chiller} is {anomaly_mean:.3f} "
            f"(positive = above expected, negative = below expected)."
        )
    if power_diff is not None:
        direction = "above" if power_diff > 0 else "below"
        bullets.append(
            f"• On average, actual power is about {abs(power_diff):.1f} kW {direction} the "
            f"predicted/design power profile."
        )

    # Recommended actions (generic but useful)
    actions = [
        "• Verify setpoints for chilled water supply and condenser water loops.",
        "• Check strainers, filters, and heat-exchanger fouling if power is trending high.",
        "• Validate sensors (temperature, flow, power meters) for drift or calibration issues.",
        "• If anomaly counts are consistently high, review KPI thresholds and alarm logic.",
        "• For units with frequent deviations, plan targeted maintenance or inspection.",
    ]

    text = [
        f"**Offline explanation for chiller {chiller}** *(no cloud model used)*\n",
        f"**User question:** {query}\n",
        "### What the data suggests:",
        *bullets,
        "",
        "### Recommended actions for operations:",
        *actions,
    ]

    return "\n".join(text)


# ---------------------------------------------------------------------
# Gemini call helper
# ---------------------------------------------------------------------
def _call_gemini(prompt: str) -> str:
    """Call Gemini model with proper error handling and fall back on quota errors."""
    if not (GEMINI_AVAILABLE and API_KEY):
        # No SDK or no key → let caller fall back
        raise RuntimeError("Gemini not available or API key missing.")

    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    response = model.generate_content(prompt)

    if hasattr(response, "text") and response.text:
        return response.text.strip()
    return str(response)


# ---------------------------------------------------------------------
# Prompt builder (same as before, but generic)
# ---------------------------------------------------------------------
def _build_prompt(query: str, df: pd.DataFrame, chiller: str) -> str:
    recent = df[df["chiller"] == chiller].sort_values("time").tail(24)

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
1. Directly answer the user's question in context of this data.
2. Explain WHY the graphs might show deviations or changes.
3. Explicitly name which parameter(s) are driving the deviation.
4. Explain impact for the operations team.
5. Suggest 3–6 concrete operational actions.

Be clear and concise.
"""


# ---------------------------------------------------------------------
# Public function called from app.py
# ---------------------------------------------------------------------
def ai_answer(query: str, df: pd.DataFrame, chiller: str) -> str:
    """
    Main entry point for the dashboard.

    It will:
    - Try Gemini (if available & allowed),
    - On any quota/429/permission or config error, fall back to offline rule-based answer.
    """
    prompt = _build_prompt(query, df, chiller)

    # Try Gemini first
    try:
        return _call_gemini(prompt)
    except Exception as e:
        # For debugging: show a small note in the UI
        st.info(f"Using offline explanation (Gemini unavailable: {e})")
        # Fall back to local, free explanation
        return _local_rule_based_answer(query, df, chiller)
