import streamlit as st
import json
import plotly.express as px
from simulator import simulate_timeseries, simulate_anomaly_summary, simulate_maintenance
from ai_agent import ai_answer

# --------------------------------------
# PAGE CONFIG
# --------------------------------------
st.set_page_config(page_title="AI-Driven Chiller Dashboard", layout="wide")

st.title("AI-Driven Chiller Health Monitoring")

# --------------------------------------
# LOAD CONFIG
# --------------------------------------
with open("config.json") as f:
    chillers = json.load(f)["chillers"]

# --------------------------------------
# SIMULATED DATA
# --------------------------------------
df = simulate_timeseries(chillers)
df_anom = simulate_anomaly_summary(chillers)
df_maint = simulate_maintenance(chillers)

# --------------------------------------
# PAGE MENU (4 dashboards)
# --------------------------------------
page = st.radio(
    "Select Dashboard",
    ["Power Consumption", "Anomaly Detection", "Predictive Maintenance", "Design Power"],
    horizontal=True
)

# --------------------------------------
# PAGE-WISE QUERY DROPDOWNS
# --------------------------------------
queries_page1 = [
    "Higher ambient temperature",
    "Increase in IT load",
    "Increase in chilled water inlet temperature"
]

queries_page2 = [
    "What exactly triggers a high anomaly score",
    "How the operations team should interpret each category",
    "What actions need to be taken for different anomaly levels"
]

queries_page3 = [
    "Which parameters the model uses (runtime, performance degradation, fleet score, etc.)",
    "How operations should decide on maintenance priority"
]

queries_page4 = [
    "Health KPIs and thresholds",
    "What conditions require immediate escalation by the operations team",
    "What triggers an internal alert from the model"
]

# --------------------------------------
# PAGE-SPECIFIC UI
# --------------------------------------

# ---------------------------------------------------------------
# PAGE 1 — POWER CONSUMPTION (like screenshot 1)
# ---------------------------------------------------------------
if page == "Power Consumption":

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_chiller = st.selectbox("Select Chiller", chillers)
    with col2:
        query = st.selectbox("Ask a question", queries_page1)

    df_sel = df[df["chiller"] == selected_chiller]

    # POWER GRAPH
    fig = px.line(df_sel, x="time", y=["power_predicted", "power_actual"],
                  labels={"value": "kW"},
                  title=f"Predicted vs Actual Power – {selected_chiller}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("AI Interpretation & Deviation Reason")
    st.write(ai_answer(query, df, selected_chiller))


# ---------------------------------------------------------------
# PAGE 2 — ANOMALY DETECTION (like screenshot 2)
# ---------------------------------------------------------------
elif page == "Anomaly Detection":

    query = st.selectbox("Ask a question", queries_page2)

    st.subheader("Anomaly Count Summary")
    fig1 = px.bar(df_anom, x="chiller", y="count")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Anomaly Score Distribution")
    fig2 = px.histogram(df, x="anomaly_score", nbins=40)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("AI Interpretation")
    st.write(ai_answer(query, df, chillers[0]))


# ---------------------------------------------------------------
# PAGE 3 — PREDICTIVE MAINTENANCE (screenshot 3)
# ---------------------------------------------------------------
elif page == "Predictive Maintenance":

    query = st.selectbox("Ask a question", queries_page3)

    st.subheader("Maintenance Due (Days)")
    fig = px.bar(df_maint, x="chiller", y="due_days", color="priority")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("AI Interpretation")
    st.write(ai_answer(query, df, chillers[0]))


# ---------------------------------------------------------------
# PAGE 4 — DESIGN POWER (screenshot 4)
# ---------------------------------------------------------------
elif page == "Design Power":

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_chiller = st.selectbox("Select Chiller", chillers)
    with col2:
        query = st.selectbox("Ask a question", queries_page4)

    df_sel = df[df["chiller"] == selected_chiller]

    fig = px.line(df_sel, x="time", y="power_actual",
                  title=f"Design Power Overlay (Simulated) – {selected_chiller}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("AI Interpretation")
    st.write(ai_answer(query, df, selected_chiller))
