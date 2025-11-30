import os
import pandas as pd
import google.generativeai as genai
import streamlit as st


# ---------------------------
# Configure Gemini client
# ---------------------------
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    # Helpful message inside the Streamlit UI
    st.warning(
        "⚠️ GOOGLE_API_KEY is not set. "
        "Go to Streamlit → Settings → Secrets and add GOOGLE_API_KEY."
    )
else:
    genai.configure(api_key=API_KEY)

# Choose a Gemini model
MODEL_NAME = "gemini-1.5-flash"


def _build_prompt(query: str, df: pd.DataFrame, chiller: str) -> str:
    """
    Build a compact prompt with the latest data for the selected chiller.
    """
    recent = df[df["chiller"] == chiller].tail(12)

    return f"""
You are an expert data-center BMS / chiller analytics assistant.

User query:
{query}

Selected chiller: {chiller}

Below is the last 12 time-steps of data for this chiller
(columns: time, ambient, it_load, chw_in, power_predicted, power_actual, anomaly_score):

{recent.to_string(index=False)}

Based on this:

1. Answer the user's question in simple clear language.
2. Explain why the graph is deviating (e.g., ambient temp higher, IT load spike, CHW inlet higher, sensor drift, etc.).
3. Explicitly mention which parameter(s) caused the deviation.
4. Describe operational impact (energy, reliability, cooling risk).
5. Recommend concrete actions for the operations team.

Keep the answer concise but structured with short bullet points.
"""


def ai_answer(query: str, df: pd.DataFrame, chiller: str) -> str:
    """
    Call Gemini to answer the query + explain graph deviations.
    Signature kept same as before so app.py does not need changes.
    """
    if not API_KEY:
        return (
            "GOOGLE_API_KEY is not configured. "
            "Please set it in Streamlit Secrets as GOOGLE_API_KEY."
        )

    prompt = _build_prompt(query, df, chiller)

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        # Some SDK versions use .text, others .candidates
        if hasattr(response, "text"):
            return response.text
        else:
            return str(response)
    except Exception as e:
        return f"❌ Gemini error: {e}"
