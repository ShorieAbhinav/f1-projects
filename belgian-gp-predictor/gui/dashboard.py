import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide"
)

# ── Sidebar author tag ─────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Built by:** Abhinav Shorie")
st.sidebar.markdown("**Stack:** FastF1 · scikit-learn · Monte Carlo")
st.sidebar.markdown("**GitHub:** [ShorieAbhinav](https://github.com/ShorieAbhinav)")

# ── Header ─────────────────────────────────────────────────────
st.title("🏎️ F1 Race Predictor")
st.subheader("2026 Belgian Grand Prix — Monte Carlo Simulation")
st.markdown("---")

# ── Load the simulation results from CSV ──────────────────────
# Unlike the Chinese GP dashboard (which ran the whole pipeline live
# on every page load), this reads the already-saved simulation CSV.
# Faster, and doesn't re-run 100k sims each time the page refreshes.
@st.cache_data
def load_results():
    path = os.path.join(os.path.dirname(__file__), "..", "monte_carlo",
                        "belgian_gp_2026_simulation.csv")
    df = pd.read_csv(path)
    return df.sort_values("WinProbability", ascending=False).reset_index(drop=True)

results = load_results()

CHAOS_COEFFICIENT = 3.44

# ── Top metric cards ───────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🏆 Predicted Winner",
        value=results.iloc[0]["FullName"],
        delta=f"{results.iloc[0]['WinProbability']*100:.1f}% win chance"
    )

with col2:
    st.metric(
        label="🥈 Predicted P2",
        value=results.iloc[1]["FullName"],
        delta=f"{results.iloc[1]['WinProbability']*100:.1f}% win chance"
    )

with col3:
    st.metric(
        label="🥉 Predicted P3",
        value=results.iloc[2]["FullName"],
        delta=f"{results.iloc[2]['WinProbability']*100:.1f}% win chance"
    )

with col4:
    st.metric(
        label="🌀 Chaos Coefficient",
        value=f"{CHAOS_COEFFICIENT:.2f}",
        delta="avg position changes at Spa"
    )

st.markdown("---")

# ── Team colors for the charts ─────────────────────────────────
# Note: your simulation CSV doesn't carry TeamName, so bars use a
# single F1-red. If you want team colors, add TeamName to the
# simulation output in simulate.py and map it here like Chinese GP did.
BAR_COLOR = "#e10600"

# ── Win probability bar chart ──────────────────────────────────
st.subheader("🏁 Win Probability by Driver")

fig_win = go.Figure(go.Bar(
    x=results["FullName"],
    y=results["WinProbability"] * 100,
    marker_color=BAR_COLOR,
    text=[f"{p*100:.1f}%" for p in results["WinProbability"]],
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>Win: %{y:.2f}%<extra></extra>"
))

fig_win.update_layout(
    plot_bgcolor="#0D1117",
    paper_bgcolor="#0D1117",
    font_color="#E6EDF3",
    xaxis=dict(tickangle=-45, gridcolor="#21262D"),
    yaxis=dict(title="Win Probability (%)", gridcolor="#21262D"),
    height=450,
    showlegend=False,
    margin=dict(t=20, b=120)
)

st.plotly_chart(fig_win, use_container_width=True)
st.markdown("---")

# ── Podium probability bar chart ───────────────────────────────
st.subheader("🏅 Podium Probability by Driver")

fig_podium = go.Figure(go.Bar(
    x=results["FullName"],
    y=results["PodiumProbability"] * 100,
    marker_color=BAR_COLOR,
    text=[f"{p*100:.1f}%" for p in results["PodiumProbability"]],
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>Podium: %{y:.2f}%<extra></extra>"
))

fig_podium.update_layout(
    plot_bgcolor="#0D1117",
    paper_bgcolor="#0D1117",
    font_color="#E6EDF3",
    xaxis=dict(tickangle=-45, gridcolor="#21262D"),
    yaxis=dict(title="Podium Probability (%)", gridcolor="#21262D"),
    height=450,
    showlegend=False,
    margin=dict(t=20, b=120)
)

st.plotly_chart(fig_podium, use_container_width=True)
st.markdown("---")

# ── Full predictions table ─────────────────────────────────────
st.subheader("📊 Full Predictions Table")

table = pd.DataFrame({
    "Rank": range(1, len(results) + 1),
    "Driver": results["FullName"],
    "Grid": [f"P{int(g)}" for g in results["GridPosition"]],
    "Predicted Finish": [f"{p:.1f}" for p in results["PredictedFinish"]],
    "Win %": [f"{p*100:.2f}%" for p in results["WinProbability"]],
    "Podium %": [f"{p*100:.2f}%" for p in results["PodiumProbability"]],
})

st.dataframe(table, use_container_width=True, hide_index=True)
st.markdown("---")

# ── Model info ─────────────────────────────────────────────────
# IMPORTANT: this describes YOUR actual model, not Chinese GP's
# heuristic weights. Yours is a trained GradientBoosting regressor,
# so the honest description is the features + MAE, not fake weights.
st.subheader("ℹ️ Model Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Prediction Model:**
    - Gradient Boosting Regressor (scikit-learn)
    - Features: grid position + team-year form
    - Trained on 2019–2025 Spa races (finishers only)
    - Test-set MAE: ~1.85 finishing positions
    """)

with col2:
    st.markdown(f"""
    **Simulation Parameters:**
    - Simulations: 100,000
    - Chaos Coefficient: {CHAOS_COEFFICIENT:.2f}
    - Race-day rain probability: 25% (Sunday forecast)
    - Wet races: chaos ×2, DNF risk ×1.5
    - Data Source: FastF1 API
    """)

st.markdown("---")
st.caption("Built by Abhinav Shorie · FastF1 + scikit-learn + Monte Carlo · 2026")