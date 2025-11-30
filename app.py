# app.py
import json

import streamlit as st
import plotly.express as px

from simulator import (
    simulate_timeseries,
    simulate_anomaly_summary,
    simulate_maintenance,
)
from ai_agent import ai_answer


# -------------------------------------------------
# Streamlit page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="AI-Driven Chiller Health Monitoring",
    layout="wide",
)


# -------------------------------------------------
# Load configuration (chiller list)
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_chillers():
    with open("config.json") as f:
        cfg = json.load(f)
    return cfg["chillers"]


# -------------------------------------------------
# Simulate data (cached)
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    chillers_local = load_chillers()
    df_ts = simulate_timeseries(chillers_local)
    df_anom = simulate_anomaly_summary(chillers_local)
    df_maint = simulate_maintenance(chillers_local)
    return df_ts, df_anom, df_maint


chillers = load_chillers()
df_ts, df_anom, df_maint = load_data()


# -------------------------------------------------
# Page-level query options
# -------------------------------------------------
queries_page1 = [
    "Higher ambient temperature",
    "Increase in IT load",
    "Increase in chilled water inlet temperature",
]

queries_page2 = [
    "What exactly triggers a high anomaly score",
    "How the operations team should interpret each category",
    "What actions need to be taken for different anomaly levels",
]

queries_page3 = [
    "Which parameters the model uses (runtime, performance degradation, fleet score, etc.)",
    "How operations should decide on maintenance priority",
]

queries_page4 = [
    "Health KPIs and thresholds",
    "What conditions require immediate escalation by the operations team",
    "What triggers an internal alert from the model",
]


# -------------------------------------------------
# Top-level layout
# -------------------------------------------------
st.markdown(
    "<h2 style='margin-bottom:0px;'>AI-Driven Chiller Health Monitoring</h2>",
    unsafe_allow_html=True,
)
st.caption("Single dashboard URL with four AI-assisted views")

page = st.radio(
    "Select dashboard mode",
    [
        "Power Consumption",
        "Anomaly Detection",
        "Predictive Maintenance",
        "Design Power",
    ],
    horizontal=True,
)


# =================================================
# PAGE 1 – POWER CONSUMPTION
# =================================================
if page == "Power Consumption":
    st.markdown("### Power Consumption – Predicted vs Actual per Chiller")

    top_l, top_r = st.columns([2, 1])

    with top_l:
        selected_chiller = st.selectbox("Select chiller", chillers, key="p1_chiller")

    with top_r:
        query = st.selectbox("Analysis question", queries_page1, key="p1_query")

    df_sel = df_ts[df_ts["chiller"] == selected_chiller]

    # Main graph: predicted vs actual power
    fig_power = px.line(
        df_sel,
        x="time",
        y=["power_predicted", "power_actual"],
        labels={"value": "Power (kW)", "time": "Time"},
        title=f"Predicted vs Actual Power – {selected_chiller}",
    )
    fig_power.update_layout(legend_title_text="Series")
    st.plotly_chart(fig_power, use_container_width=True)

    # Supporting distributions (optional – looks like your reference UI)
    c1, c2 = st.columns(2)
    with c1:
        fig_ambient = px.line(
            df_sel,
            x="time",
            y="ambient",
            labels={"ambient": "Ambient Temperature (°C)", "time": "Time"},
            title="Ambient Temperature Profile",
        )
        st.plotly_chart(fig_ambient, use_container_width=True)
    with c2:
        fig_it = px.line(
            df_sel,
            x="time",
            y="it_load",
            labels={"it_load": "IT Load (%)", "time": "Time"},
            title="IT Load Profile",
        )
        st.plotly_chart(fig_it, use_container_width=True)

    st.markdown("#### AI explanation & deviation reason")
    explanation = ai_answer(query, df_ts, selected_chiller)
    st.write(explanation)


# =================================================
# PAGE 2 – ANOMALY DETECTION
# =================================================
elif page == "Anomaly Detection":
    st.markdown("### Anomaly Detection Summary")

    query = st.selectbox("Analysis question", queries_page2, key="p2_query")

    # 1) Anomaly count per chiller (bar)
    st.markdown("#### Anomaly count per chiller (last 7 days – simulated)")
    fig_count = px.bar(
        df_anom,
        x="chiller",
        y="count",
        labels={"chiller": "Chiller", "count": "Anomaly count"},
    )
    st.plotly_chart(fig_count, use_container_width=True)

    # 2) Anomaly score distribution (histogram)
    st.markdown("#### Anomaly score distribution")
    fig_hist = px.histogram(
        df_ts,
        x="anomaly_score",
        nbins=40,
        labels={"anomaly_score": "Anomaly score"},
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # 3) Per-chiller anomaly trend (line)
    st.markdown("#### Mean anomaly score per chiller (trend)")
    df_mean = (
        df_ts.groupby(["time", "chiller"])["anomaly_score"]
        .mean()
        .reset_index()
        .sort_values("time")
    )
    fig_trend = px.line(
        df_mean,
        x="time",
        y="anomaly_score",
        color="chiller",
        labels={"anomaly_score": "Mean anomaly score"},
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("#### AI interpretation of anomaly behaviour")
    # For anomaly-level questions, we can pick an arbitrary 'fleet' anchor chiller
    explanation = ai_answer(query, df_ts, chillers[0])
    st.write(explanation)


# =================================================
# PAGE 3 – PREDICTIVE MAINTENANCE
# =================================================
elif page == "Predictive Maintenance":
    st.markdown("### Predictive Maintenance Dashboard")

    query = st.selectbox("Analysis question", queries_page3, key="p3_query")

    # Bar of days until maintenance due, coloured by priority
    st.markdown("#### Maintenance priority by chiller")
    fig_maint = px.bar(
        df_maint,
        x="chiller",
        y="due_days",
        color="priority",
        labels={
            "chiller": "Chiller",
            "due_days": "Days until next maintenance (simulated)",
            "priority": "Priority",
        },
    )
    st.plotly_chart(fig_maint, use_container_width=True)

    # Scatter timeline (x = days until maintenance, y = chiller)
    st.markdown("#### Maintenance timeline (days until next job)")
    fig_timeline = px.scatter(
        df_maint,
        x="due_days",
        y="chiller",
        size=[8] * len(df_maint),
        color="priority",
        labels={"due_days": "Days until next maintenance", "chiller": "Chiller"},
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown("#### AI explanation of maintenance drivers & priorities")
    explanation = ai_answer(query, df_ts, chillers[0])
    st.write(explanation)


# =================================================
# PAGE 4 – DESIGN POWER / HEALTH KPIs
# =================================================
elif page == "Design Power":
    st.markdown("### Design Power & Health KPIs")

    top_l, top_r = st.columns([2, 1])

    with top_l:
        selected_chiller = st.selectbox("Select chiller", chillers, key="p4_chiller")

    with top_r:
        query = st.selectbox("Analysis question", queries_page4, key="p4_query")

    df_sel = df_ts[df_ts["chiller"] == selected_chiller]

    # Here we treat predicted power as 'design' envelope and actual as real operation
    st.markdown("#### Design vs actual operating profile")
    fig_design = px.line(
        df_sel,
        x="time",
        y=["power_predicted", "power_actual"],
        labels={"value": "Power (kW)", "time": "Time"},
        title=f"Design (predicted) vs Actual Power – {selected_chiller}",
    )
    st.plotly_chart(fig_design, use_container_width=True)

    # Simple KPI summary (simulated) – similar feel to your reference cards
    st.markdown("#### Simulated health KPIs")
    kpi_1, kpi_2, kpi_3 = st.columns(3)
    with kpi_1:
        st.metric(
            "Avg efficiency index",
            f"{(df_sel['power_predicted'].sum() / max(df_sel['power_actual'].sum(), 1)):.2f}",
        )
    with kpi_2:
        st.metric(
            "Avg anomaly score",
            f"{df_sel['anomaly_score'].abs().mean():.3f}",
        )
    with kpi_3:
        st.metric(
            "Hours analysed",
            f"{len(df_sel) * 0.25:.1f} h",  # 15-min samples → 0.25h
        )

    st.markdown("#### AI explanation: KPIs, thresholds, alerts & escalation")
    explanation = ai_answer(query, df_ts, selected_chiller)
    st.write(explanation)
