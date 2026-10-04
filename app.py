"""
app.py — Virat Kohli 100th International Century Predictor

Run with:
    streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from predictor import (
    SimulationResult,
    estimate_century_prob_from_data,
    load_innings_data,
    run_simulation,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Virat Kohli — 100th Century Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Minimal cricket-themed CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Background */
    .stApp { background-color: #0a1628; color: #e8e8e8; }
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #112240;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 12px 16px;
    }
    /* Section headers */
    .section-header {
        color: #f5a623;
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 4px;
    }
    /* Probability badge */
    .prob-badge {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    .prob-high { background: #0d3d1a; color: #4caf50; border: 2px solid #4caf50; }
    .prob-mid  { background: #3d2d00; color: #f5a623; border: 2px solid #f5a623; }
    .prob-low  { background: #3d0d0d; color: #ef5350; border: 2px solid #ef5350; }
    /* Plotly chart background */
    .js-plotly-plot .plotly .bg { fill: #112240 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "data" / "innings_data.csv"
TEMPLATE_PATH = Path(__file__).parent / "data" / "innings_template.csv"

DEFAULT_CURRENT = 86
DEFAULT_TARGET = 100
DEFAULT_INNINGS = 150
DEFAULT_SIMULATIONS = 10_000
SEED = 42

# Historical estimated probability (Kohli's career century rate across formats)
# Based on publicly known career statistics:
#   ~86 centuries in ~550 international innings (approx.)
# This is a pre-loaded estimate; actual historical CSV can refine it.
HISTORICAL_ESTIMATE = 0.157   # ~15.7 % per innings

# Scenario probability defaults
SCENARIO_PROBS = {
    "Conservative": 0.10,
    "Base Case":    HISTORICAL_ESTIMATE,
    "Optimistic":   0.20,
}

# ---------------------------------------------------------------------------
# Load historical data (optional)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def try_load_data():
    df = load_innings_data(DATA_PATH)
    if df is not None:
        prob, meta = estimate_century_prob_from_data(df)
        return df, prob, meta
    return None, None, None


innings_df, hist_prob, hist_meta = try_load_data()
data_loaded = innings_df is not None

# If real data is loaded, use that probability as the base-case default
if data_loaded and hist_prob:
    SCENARIO_PROBS["Base Case"] = hist_prob

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <h1 style='text-align:center; color:#f5a623; margin-bottom:0;'>
        🏏 Virat Kohli — 100th Century Predictor
    </h1>
    <p style='text-align:center; color:#8899aa; margin-top:4px;'>
        Monte Carlo simulation · International centuries only (Test + ODI + T20I)
    </p>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Data status banner
# ---------------------------------------------------------------------------

if data_loaded:
    st.success(
        f"✅ Historical innings data loaded — {hist_meta['total_innings']} innings, "
        f"{hist_meta['century_innings']} centuries. "
        f"Estimated rate: **{hist_prob:.1%}** per innings."
    )
else:
    st.info(
        "ℹ️ No historical innings CSV found at `data/innings_data.csv`. "
        "Using manually configured probability assumptions. "
        "See `data/README.md` for instructions on adding real data."
    )

# ---------------------------------------------------------------------------
# Input panel
# ---------------------------------------------------------------------------

st.markdown("<div class='section-header'>📋 Inputs</div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    current_centuries = st.number_input(
        "Current centuries",
        min_value=0, max_value=200,
        value=DEFAULT_CURRENT,
        step=1,
        help="Kohli's confirmed international century count.",
    )

with col2:
    target = st.number_input(
        "Target centuries",
        min_value=1, max_value=500,
        value=DEFAULT_TARGET,
        step=1,
        help="The milestone to predict.",
    )

with col3:
    remaining_innings = st.number_input(
        "Remaining international innings",
        min_value=0, max_value=1000,
        value=DEFAULT_INNINGS,
        step=10,
        help="Estimated innings still to be played.",
    )

with col4:
    n_simulations = st.select_slider(
        "Simulation runs",
        options=[1_000, 5_000, 10_000, 50_000, 100_000],
        value=DEFAULT_SIMULATIONS,
        help="More runs = smoother results, slower computation.",
    )

gap = target - current_centuries
if gap <= 0:
    st.success(f"🎉 Target already reached! ({current_centuries} ≥ {target})")

# ---------------------------------------------------------------------------
# Scenario probability controls
# ---------------------------------------------------------------------------

st.markdown("<div class='section-header'>🎯 Scenario Assumptions</div>", unsafe_allow_html=True)

st.caption(
    "These are *scenario assumptions*, not guaranteed forecasts. "
    "Adjust them to explore different outlooks."
)

scol1, scol2, scol3 = st.columns(3)

with scol1:
    prob_conservative = st.slider(
        "Conservative probability / innings",
        min_value=0.01, max_value=0.50,
        value=SCENARIO_PROBS["Conservative"],
        step=0.005,
        format="%.3f",
        help="Pessimistic scenario — lower form or fewer innings.",
    )

with scol2:
    prob_base = st.slider(
        "Base-case probability / innings",
        min_value=0.01, max_value=0.50,
        value=SCENARIO_PROBS["Base Case"],
        step=0.005,
        format="%.3f",
        help=(
            "Based on historical data." if data_loaded
            else "Manual assumption — no data file loaded."
        ),
    )

with scol3:
    prob_optimistic = st.slider(
        "Optimistic probability / innings",
        min_value=0.01, max_value=0.50,
        value=SCENARIO_PROBS["Optimistic"],
        step=0.005,
        format="%.3f",
        help="Optimistic scenario — peak form sustained.",
    )

# ---------------------------------------------------------------------------
# Run simulation on button press (or immediately if target reached)
# ---------------------------------------------------------------------------

predict_col, _ = st.columns([1, 3])
with predict_col:
    run_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

if run_btn or gap <= 0:
    scenarios = [
        ("Conservative", prob_conservative),
        ("Base Case",    prob_base),
        ("Optimistic",   prob_optimistic),
    ]

    results: dict[str, SimulationResult] = {}
    for name, prob in scenarios:
        results[name] = run_simulation(
            current_centuries=int(current_centuries),
            target=int(target),
            remaining_innings=int(remaining_innings),
            century_prob=prob,
            n_simulations=n_simulations,
            seed=SEED,
            scenario_name=name,
        )

    base = results["Base Case"]

    # -----------------------------------------------------------------------
    # Primary result
    # -----------------------------------------------------------------------

    st.divider()
    st.markdown("<div class='section-header'>📊 Primary Result — Base Case</div>", unsafe_allow_html=True)

    pct = base.prob_reach_target
    badge_class = "prob-high" if pct >= 0.6 else ("prob-mid" if pct >= 0.35 else "prob-low")
    verdict = "✅ Likely to reach target" if pct >= 0.5 else "❌ Unlikely to reach target"

    rc1, rc2 = st.columns([1, 2])
    with rc1:
        st.markdown(
            f"<div class='prob-badge {badge_class}'>{pct:.1%}</div>"
            f"<p style='text-align:center; color:#8899aa;'>{verdict}</p>",
            unsafe_allow_html=True,
        )

    with rc2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Need", f"{max(gap, 0)} more")
        m2.metric("Expected total", f"{base.expected_final_total:.1f}")
        m3.metric("Median total",   f"{base.median_final_total:.0f}")
        m4.metric("Expected extra", f"+{base.expected_additional:.1f}")

        st.caption(
            f"10th pct: **{base.p10_final_total:.0f}** · "
            f"90th pct: **{base.p90_final_total:.0f}** · "
            f"Century prob/innings: **{prob_base:.1%}** · "
            f"Simulations: **{n_simulations:,}** · Seed: **{SEED}**"
        )

    # -----------------------------------------------------------------------
    # Histogram of simulated final totals (base case)
    # -----------------------------------------------------------------------

    st.markdown("<div class='section-header'>📈 Distribution of Simulated Final Century Totals</div>", unsafe_allow_html=True)

    totals = base.final_totals
    bin_min = int(totals.min())
    bin_max = int(totals.max())

    # Build histogram data
    counts, bin_edges = np.histogram(totals, bins=min(50, bin_max - bin_min + 1))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    colors = [
        "#4caf50" if c >= target else "#ef5350"
        for c in bin_centers
    ]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(
        x=bin_centers,
        y=counts,
        marker_color=colors,
        name="Simulations",
        hovertemplate="Final total: %{x:.0f}<br>Count: %{y}<extra></extra>",
    ))
    fig_hist.add_vline(
        x=target,
        line_dash="dash",
        line_color="#f5a623",
        annotation_text=f"Target ({target})",
        annotation_font_color="#f5a623",
    )
    fig_hist.add_vline(
        x=current_centuries,
        line_dash="dot",
        line_color="#8899aa",
        annotation_text=f"Current ({current_centuries})",
        annotation_font_color="#8899aa",
        annotation_position="top left",
    )
    fig_hist.update_layout(
        paper_bgcolor="#112240",
        plot_bgcolor="#0d1e36",
        font_color="#e8e8e8",
        xaxis_title="Final Century Total",
        yaxis_title="Number of Simulations",
        bargap=0.05,
        showlegend=False,
        margin=dict(t=30, b=40, l=50, r=20),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.caption(
        "Green bars = simulations where Kohli reaches the target. "
        "Red bars = simulations where he falls short."
    )

    # -----------------------------------------------------------------------
    # Scenario comparison
    # -----------------------------------------------------------------------

    st.markdown("<div class='section-header'>🔄 Scenario Comparison</div>", unsafe_allow_html=True)

    scenario_names  = list(results.keys())
    scenario_probs  = [results[n].prob_reach_target for n in scenario_names]
    scenario_colors = ["#ef5350", "#f5a623", "#4caf50"]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=scenario_names,
        y=[p * 100 for p in scenario_probs],
        marker_color=scenario_colors,
        text=[f"{p:.1%}" for p in scenario_probs],
        textposition="outside",
        textfont_color="#e8e8e8",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    fig_bar.add_hline(
        y=50,
        line_dash="dash",
        line_color="#ffffff",
        opacity=0.4,
        annotation_text="50 % threshold",
        annotation_font_color="#8899aa",
    )
    fig_bar.update_layout(
        paper_bgcolor="#112240",
        plot_bgcolor="#0d1e36",
        font_color="#e8e8e8",
        yaxis_title="Probability of Reaching Target (%)",
        yaxis_range=[0, 110],
        showlegend=False,
        margin=dict(t=30, b=40, l=50, r=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Scenario summary table
    scenario_table = pd.DataFrame({
        "Scenario":          scenario_names,
        "Prob / innings":    [f"{p:.1%}" for p in [prob_conservative, prob_base, prob_optimistic]],
        "P(reach target)":   [f"{results[n].prob_reach_target:.1%}" for n in scenario_names],
        "Expected total":    [f"{results[n].expected_final_total:.1f}" for n in scenario_names],
        "Median total":      [f"{results[n].median_final_total:.0f}" for n in scenario_names],
        "10th pct":          [f"{results[n].p10_final_total:.0f}" for n in scenario_names],
        "90th pct":          [f"{results[n].p90_final_total:.0f}" for n in scenario_names],
    })
    st.dataframe(scenario_table, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Model assumptions & limitations
    # -----------------------------------------------------------------------

    st.divider()
    st.markdown("<div class='section-header'>⚠️ Model Assumptions & Limitations</div>", unsafe_allow_html=True)

    with st.expander("Read before interpreting results", expanded=False):
        st.markdown(f"""
**What this model does**
- Runs {n_simulations:,} independent Monte Carlo simulations.
- Each future innings is modelled as a Bernoulli trial: century with probability *p*, no century with probability *1−p*.
- All innings are treated as independent and identically distributed (i.i.d.) — the model does not account for streaks, form cycles, or opponent quality.

**Probability source**
{"- Century probability estimated from the loaded historical innings CSV." if data_loaded else "- **No data file loaded.** Century probability is a user-configured assumption only."}
- Historical estimate used as Base Case: **{prob_base:.3f}** ({prob_base:.1%} per innings).
- Kohli's published approximate career statistics (pre-2024): ~86 centuries in ~550 international innings, implying ~15.7 %.

**Known limitations**
- The model cannot account for injury, retirement, or future selection decisions.
- Probability per innings is assumed constant — real performance varies with age, opposition, and format.
- "International innings" is user-defined; the default ({remaining_innings}) is a rough assumption.
- T20I centuries are extremely rare; the combined format probability is dominated by Test and ODI data.
- This is a *scenario tool*, not a calibrated forecast. Results should be read as "if assumptions hold, then…" statements.
- IPL or domestic centuries are **not** counted; only Test, ODI, and T20I centuries are international.

**Reproducibility**
- Random seed fixed at **{SEED}**. Re-running with the same inputs produces identical results.
        """)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.markdown(
    "<p style='text-align:center; color:#445566; font-size:0.8rem;'>"
    "Virat Kohli 100th Century Predictor · Monte Carlo simulation · "
    "For educational and entertainment purposes only · Not financial, sporting, or betting advice."
    "</p>",
    unsafe_allow_html=True,
)
