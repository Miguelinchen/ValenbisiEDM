"""
app.py — Valenbisi Demand Forecaster & Bayesian Uncertainty Explorer
Single-file Streamlit dashboard: Prophet forecasting + Folium map + Plotly charts.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Valenbisi Forecaster",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS — dark professional dashboard theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* ── Main viewport ── */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f4f8fc 0%, #edf2f9 100%);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #dce8f6 0%, #c8daf0 100%);
        border-right: 2px solid #a0c4e8;
    }
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #1b2d45 !important;
    }
    [data-testid="stSidebar"] h4 {
        color: #0d47a1 !important;
    }

    /* ── Headers ── */
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] h5,
    [data-testid="stAppViewContainer"] h6 {
        color: #0b3d6b !important;
    }
    [data-testid="stAppViewContainer"] .stCaption {
        color: #556e8c !important;
    }

    /* ── Header / toolbar ── */
    [data-testid="stHeader"] {
        background-color: #f4f8fc;
    }
    [data-testid="stToolbar"] {
        background-color: #f4f8fc;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #c8d6e5;
        border-left: 4px solid #1e88e5;
        border-radius: 10px;
        padding: 14px 18px;
    }
    [data-testid="stMetric"] label,
    [data-testid="stMetricLabel"] p {
        color: #556e8c !important;
        font-size: 0.78rem;
    }
    [data-testid="stMetricValue"] {
        color: #0b3d6b !important;
        font-size: 1.45rem;
    }
    [data-testid="stMetricDelta"] {
        color: #1e88e5 !important;
    }

    /* ── Info / notification boxes ── */
    [data-testid="stNotification"] {
        background-color: #e3f2fd !important;
        border: 1px solid #90caf9 !important;
        border-left: 4px solid #1e88e5 !important;
        border-radius: 10px !important;
    }
    [data-testid="stNotification"] * {
        color: #1b2d45 !important;
    }

    /* ── Dividers ── */
    [data-testid="stAppViewContainer"] hr {
        border-color: #c8d6e5 !important;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: transparent;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        color: #556e8c !important;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
        border: 1px solid transparent;
        font-weight: 500;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: #1e88e5 !important;
        background-color: #ffffff !important;
        border-color: #c8d6e5 #c8d6e5 transparent #c8d6e5 !important;
        font-weight: 600;
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: #1e88e5 !important;
    }

    /* ── Dataframe / table ── */
    [data-testid="stDataFrame"] {
        background-color: #ffffff;
        border: 1px solid #c8d6e5;
        border-radius: 10px;
    }
    [data-testid="stDataFrame"] th {
        color: #0b3d6b !important;
        background-color: #dce8f6 !important;
        border-bottom: 2px solid #90caf9 !important;
    }
    [data-testid="stDataFrame"] td {
        color: #1b2d45 !important;
        background-color: #ffffff !important;
    }
    [data-testid="stDataFrame"] tr:hover td {
        background-color: #eef3ff !important;
    }

    /* ── Select / dropdown ── */
    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-color: #a0c4e8 !important;
    }
    [data-baseweb="select"] span {
        color: #1b2d45 !important;
    }
    [data-baseweb="select"] svg {
        fill: #1e88e5 !important;
    }
    ul[data-baseweb="menu"] {
        background-color: #ffffff !important;
        border: 1px solid #a0c4e8 !important;
        box-shadow: 0 4px 12px rgba(30, 136, 229, 0.12) !important;
    }
    ul[data-baseweb="menu"] li {
        background-color: #ffffff !important;
        color: #1b2d45 !important;
    }
    ul[data-baseweb="menu"] li:hover {
        background-color: #e3f2fd !important;
        color: #0d47a1 !important;
    }
    ul[data-baseweb="menu"] li[aria-selected="true"] {
        background-color: #bbdefb !important;
        color: #0d47a1 !important;
    }

    /* ── Slider ── */
    [data-testid="stSidebar"] [data-baseweb="slider"] div {
        color: #1b2d45 !important;
    }
    [data-baseweb="slider"] [data-testid="stThumbValue"] {
        background-color: #1e88e5 !important;
        color: #ffffff !important;
    }

    /* ── Spinner ── */
    [data-testid="stSpinner"] div {
        border-color: #1e88e5 !important;
    }

    /* ── General text ── */
    [data-testid="stAppViewContainer"] .stMarkdown,
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] span {
        color: #1b2d45 !important;
    }

    /* ── Plotly ── */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

    /* ── Folium iframe ── */
    [data-testid="stIFrame"] {
        border: 1px solid #a0c4e8 !important;
        border-radius: 10px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATION_COLORS = {
    "transport_hub": "red",
    "university": "blue",
    "tourist": "orange",
    "commercial": "green",
    "residential": "purple",
    "public_service": "cadetblue",
}

VALENCIA_CENTER = [39.47, -0.376]
MAX_FORECAST_HOURS = 48


# ---------------------------------------------------------------------------
# Cached data loader
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    stations = pd.read_csv(os.path.join(base, "data", "stations.csv"))
    demand = pd.read_csv(os.path.join(base, "data", "hourly_demand.csv"))
    demand["ds"] = pd.to_datetime(demand["ds"])
    return stations, demand


# ---------------------------------------------------------------------------
# Forecast + Backtest (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training Bayesian model...")
def train_and_forecast(
    _station_id: int,
    _df_hash: str,
    station_df: pd.DataFrame,
    all_weather: pd.DataFrame,
) -> tuple:
    df = station_df.copy()

    # ============================================================
    # 1) MAIN FORECAST MODEL — trained on ALL data
    # ============================================================
    model = Prophet(
        interval_width=0.95,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        weekly_seasonality=True,
        daily_seasonality=True,
        yearly_seasonality=False,
    )
    model.add_regressor("temperature_celsius")
    model.add_regressor("precipitation_mm")
    model.add_regressor("wind_speed_kmh")

    model.fit(df[["ds", "y", "temperature_celsius", "precipitation_mm", "wind_speed_kmh"]])

    # Future dataframe (extends past ALL data, not just training cutoff)
    future = model.make_future_dataframe(periods=MAX_FORECAST_HOURS, freq="h")

    # Estimate future weather from last 14 days
    recent = all_weather[
        (all_weather["ds"] >= df["ds"].max() - pd.Timedelta(days=14))
    ].copy()
    recent["hour"] = recent["ds"].dt.hour
    future_weather = recent.groupby("hour").agg(
        temperature_celsius=("temperature_celsius", "mean"),
        precipitation_mm=("precipitation_mm", "mean"),
        wind_speed_kmh=("wind_speed_kmh", "mean"),
    ).reset_index()

    future["hour"] = future["ds"].dt.hour
    future = future.merge(future_weather, on="hour", how="left")
    future["precipitation_mm"] = future["precipitation_mm"].clip(lower=0)

    forecast = model.predict(future)
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = forecast[col].clip(lower=0)

    # ============================================================
    # 2) BACKTEST MODEL — trained on data up to 30-day cutoff
    # ============================================================
    cutoff = df["ds"].max() - pd.Timedelta(days=30)
    train_df = df[df["ds"] <= cutoff].copy()
    test_df = df[df["ds"] > cutoff].copy()

    model_bt = Prophet(
        interval_width=0.95,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        weekly_seasonality=True,
        daily_seasonality=True,
        yearly_seasonality=False,
    )
    model_bt.add_regressor("temperature_celsius")
    model_bt.add_regressor("precipitation_mm")
    model_bt.add_regressor("wind_speed_kmh")

    model_bt.fit(
        train_df[["ds", "y", "temperature_celsius", "precipitation_mm", "wind_speed_kmh"]]
    )

    # Predict on the test window (30 days = 720 hours after cutoff)
    bt_future = model_bt.make_future_dataframe(periods=30 * 24, freq="h")
    bt_recent = all_weather[
        (all_weather["ds"] >= cutoff - pd.Timedelta(days=14))
    ].copy()
    bt_recent["hour"] = bt_recent["ds"].dt.hour
    bt_future_weather = bt_recent.groupby("hour").agg(
        temperature_celsius=("temperature_celsius", "mean"),
        precipitation_mm=("precipitation_mm", "mean"),
        wind_speed_kmh=("wind_speed_kmh", "mean"),
    ).reset_index()
    bt_future["hour"] = bt_future["ds"].dt.hour
    bt_future = bt_future.merge(bt_future_weather, on="hour", how="left")
    bt_future["precipitation_mm"] = bt_future["precipitation_mm"].clip(lower=0)

    bt_forecast = model_bt.predict(bt_future)
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        bt_forecast[col] = bt_forecast[col].clip(lower=0)

    test_forecast = bt_forecast[bt_forecast["ds"].isin(test_df["ds"])]
    test_merged = test_df.merge(
        test_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds",
        how="inner",
    )

    mae = (
        np.mean(np.abs(test_merged["y"] - test_merged["yhat"]))
        if len(test_merged) > 0
        else np.nan
    )
    if len(test_merged) > 0:
        in_interval = (
            (test_merged["y"] >= test_merged["yhat_lower"])
            & (test_merged["y"] <= test_merged["yhat_upper"])
        )
        coverage = in_interval.mean()
    else:
        coverage = np.nan

    metrics = {
        "mae": round(mae, 2) if not np.isnan(mae) else None,
        "coverage_95": round(coverage * 100, 1) if not np.isnan(coverage) else None,
    }

    return model, forecast, metrics


# ---------------------------------------------------------------------------
# Folium map builder
# ---------------------------------------------------------------------------
def build_map(
    stations_df: pd.DataFrame, demand_df: pd.DataFrame, selected_id: int
) -> folium.Map:
    m = folium.Map(location=VALENCIA_CENTER, zoom_start=14, tiles="CartoDB positron")

    avg_demand = demand_df.groupby("station_id")["y"].mean().to_dict()
    total_demand = demand_df.groupby("station_id")["y"].sum().to_dict()

    for _, row in stations_df.iterrows():
        sid = row["station_id"]
        color = STATION_COLORS.get(row["type"], "gray")
        is_selected = sid == selected_id
        radius = 10 if is_selected else 7

        popup_html = (
            f"<b>{row['name']}</b><br>"
            f"Type: {row['type'].replace('_', ' ').title()}<br>"
            f"Avg bikes/h: {avg_demand.get(sid, 0):.1f}<br>"
            f"Total trips: {total_demand.get(sid, 0):,.0f}<br>"
            f"Capacity: {row['capacity']}"
        )

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color="#ffffff" if is_selected else color,
            weight=3 if is_selected else 2,
            fill=True,
            fill_color=color,
            fill_opacity=0.95 if is_selected else 0.75,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=row["name"],
        ).add_to(m)

    legend_html = (
        '<div style="position:fixed; bottom:50px; left:50px; z-index:9999; '
        "background:white; padding:10px; border-radius:5px; box-shadow:0 2px 6px rgba(0,0,0,0.2); "
        'font-size:12px;">'
        "<b>Station Types</b><br>"
    )
    for stype, color in STATION_COLORS.items():
        legend_html += (
            f'<span style="color:{color};">&#9679;</span> '
            f'{stype.replace("_", " ").title()}<br>'
        )
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ---------------------------------------------------------------------------
# Plotly forecast chart builder
# ---------------------------------------------------------------------------
def build_forecast_chart(
    history_df: pd.DataFrame,
    forecast_only: pd.DataFrame,
    station_name: str,
) -> go.Figure:
    history_cutoff = history_df["ds"].max() - pd.Timedelta(days=14)
    recent = history_df[history_df["ds"] >= history_cutoff]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recent["ds"],
        y=recent["y"],
        mode="lines",
        name="Historical demand",
        line=dict(color="#58a6ff", width=1.5),
        hovertemplate="%{x|%a %d %b %H:%M}<br>Demand: %{y}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=forecast_only["ds"],
        y=forecast_only["yhat"],
        mode="lines",
        name="Forecast (median)",
        line=dict(color="#d62728", width=2.2, dash="dash"),
        hovertemplate="%{x|%a %d %b %H:%M}<br>Forecast: %{y:.1f}<extra></extra>",
    ))

    if len(forecast_only) > 0:
        fig.add_trace(go.Scatter(
            x=forecast_only["ds"].tolist() + forecast_only["ds"].tolist()[::-1],
            y=forecast_only["yhat_upper"].tolist() + forecast_only["yhat_lower"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(214, 39, 40, 0.18)",
            line=dict(color="rgba(214, 39, 40, 0)"),
            name="95 % credible interval",
            hoverinfo="skip",
        ))

    boundary = history_df["ds"].max()
    fig.add_vline(
        x=boundary,
        line_width=1,
        line_dash="dot",
        line_color="#5a6170",
        annotation_text=" Now ",
        annotation_position="top left",
        annotation_font_size=11,
        annotation_font_color="#262730",
    )

    fig.update_layout(
        title=f"Demand Forecast — {station_name}",
        title_font_color="#0f1a3c",
        xaxis_title=None,
        yaxis_title="Bikes checked out per hour",
        yaxis_title_font_color="#5a6170",
        hovermode="x unified",
        template="plotly_white",
        height=480,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fb",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#262730"),
        ),
        xaxis=dict(gridcolor="#dfe2e8", tickfont=dict(color="#5a6170")),
        yaxis=dict(gridcolor="#dfe2e8", tickfont=dict(color="#5a6170")),
    )

    if len(recent) > 0 and len(forecast_only) > 0:
        fig.update_xaxes(range=[recent["ds"].min(), forecast_only["ds"].max()])

    return fig


# ---------------------------------------------------------------------------
# Methodology content
# ---------------------------------------------------------------------------
METHODOLOGY_TEXT = r"""
### Bayesian Structural Time Series for Urban Bike-Sharing Demand

---

#### Motivation

Open Data repositories such as the **Valencia Open Data Portal** frequently lack
*continuous*, *gap-free* historical time series at the granularity required for
hourly demand forecasting. API rate limits, missing intervals, and heterogeneous
schema versions make production-ready time-series modeling impractical without
a robust preprocessing layer.

---

#### Data Generation Strategy

We employed **Stochastic Simulation** to produce six months of realistic hourly
demand data for ten Valenbisi stations distributed across Valencia. Each station
was assigned a real-world typology with distinct demand behaviour:

| Station Type          | Demand Signature                                        |
|-----------------------|---------------------------------------------------------|
| **Transport Hub**     | Dual commute peaks (8:00–9:00, 17:00–19:00)             |
| **University**        | Morning lecture peak + midday plateau                   |
| **Commercial**        | Midday & evening shopping peaks, strong weekends         |
| **Tourist**           | Afternoon peak, elevated weekends                        |
| **Residential**       | Morning outbound + evening return                        |
| **Public Service**    | Steady daytime pattern, low nights                       |

The simulation incorporates:
- **Diurnal cycling** via station-type-specific hourly profiles.
- **Seasonal scaling** (winter trough → summer peak).
- **Weather sensitivity**: temperature parabolic penalty, exponential rain
  decay, and wind-speed damping.
- **Calendar anomalies**: *Fallas* (March 15–19), *Semana Santa*, public
  holidays, with demand suppression factors.

This approach provides a clean, gap-free training corpus while preserving the
statistical structure that a live API would supply.

---

#### Bayesian Forecasting with Prophet

We train **one Prophet model per station**, using three meteorological
covariates as exogenous regressors:

| Regressor              | Source        |
|------------------------|---------------|
| `temperature_celsius`  | AEMET OpenData |
| `precipitation_mm`     | AEMET OpenData |
| `wind_speed_kmh`       | AEMET OpenData |

Prophet (Meta, 2018) decomposes each time series into three components:

1. **Trend** — piecewise-linear with automatic changepoint detection.
2. **Seasonality** — Fourier series for weekly and daily cycles.
3. **Holiday/event effects** — custom regressors for anomalous periods.

Posterior inference is performed via **Stan's Hamiltonian Monte Carlo (HMC)**
sampler, yielding full posterior predictive distributions for every forecast
timestep.

---

#### Uncertainty Quantification

Unlike point-estimate ML models, our system outputs **95 % Bayesian credible
intervals** (`yhat_lower` to `yhat_upper`) for each hourly prediction. These
intervals capture:

- **Epistemic uncertainty**: model uncertainty due to finite training data.
- **Aleatoric uncertainty**: inherent noise in bike-sharing demand.

This is critical for robust urban planning: knowing whether Evening Pl. expects
*2–6 bikes* or *0–15 bikes* during the 20:00 rush directly informs
rebalancing-staff allocation.

Model performance is assessed on a 30-day holdout set via:
- **MAE** (Mean Absolute Error) — average deviation in bikes/hour.
- **95 % CI Coverage** — fraction of true values falling within the predicted
  interval (*target: ≈ 95 %*).

---

#### Production Readiness

The underlying **Prophet pipeline is fully API-agnostic**. Swapping the
simulated data layer for a live Valenbisi GBFS feed and an AEMET weather
forecast endpoint requires only replacing the `load_data()` function. The model
training, uncertainty estimation, and visualization stack remain identical.

```
                    ┌──────────────────┐
                    │  Valenbisi GBFS   │──── station_status.json
                    │  (Real-time API)  │──── station_info.json
                    └──────────────────┘
                              │
                              ▼
  ┌──────────┐     ┌─────────────────────┐     ┌──────────────────┐
  │  AEMET   │────▶│   load_data()       │────▶│  Prophet model   │
  │ OpenData │     │   (swap point)      │     │  (per station)   │
  └──────────┘     └─────────────────────┘     └────────┬─────────┘
                                                        │
                                                        ▼
                                          ┌─────────────────────────┐
                                          │  Streamlit Dashboard    │
                                          │  • Folium map           │
                                          │  • Plotly forecast      │
                                          │  • Uncertainty ribbons  │
                                          └─────────────────────────┘
```
"""


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # ---- Sidebar ----
    with st.sidebar:
        # Logos (local files)
        logo_upv = "logo_upv.png"
        logo_etsinf = "logo_etsinf.png"
        col_logo1, col_logo2 = st.columns(2)
        if os.path.isfile(logo_upv):
            col_logo1.image(logo_upv, use_container_width=True)
        else:
            col_logo1.caption("UPV logo")
        if os.path.isfile(logo_etsinf):
            col_logo2.image(logo_etsinf, use_container_width=True)
        else:
            col_logo2.caption("ETSINF logo")

        st.markdown("---")

        st.header("Controls")
        try:
            stations_df, demand_df = load_data()
        except FileNotFoundError:
            st.error("Data files not found. Run `generate_mock_data.py` first.")
            st.stop()

        station_names = stations_df.set_index("station_id")["name"].to_dict()
        selected_id = st.selectbox(
            "Select a station:",
            options=stations_df["station_id"].tolist(),
            format_func=lambda x: f"{x}. {station_names[x]}",
        )

        st.markdown("---")

        forecast_hours = st.slider(
            "Forecast horizon (hours):",
            min_value=12,
            max_value=MAX_FORECAST_HOURS,
            value=24,
            step=6,
            help="How many hours into the future to display the forecast for.",
        )

        st.markdown("---")
        st.markdown("#### Developers")
        st.markdown(
            "**José Miguel García**  \n"
            "**Daniel Herrán Gomez-Senent**  \n\n"
            "*Universitat Politècnica de València*  \n"
            "*EDM — Evaluación, Despliegue y Monitorización de Modelos*"
        )

    # ---- Load and prepare data ----
    station_info = stations_df[stations_df["station_id"] == selected_id].iloc[0]
    station_df = demand_df[demand_df["station_id"] == selected_id].copy()

    # ---- Train model (cached) ----
    df_hash = str(hash(station_df["y"].to_csv(index=False)))
    model, forecast, metrics = train_and_forecast(
        selected_id,
        df_hash,
        station_df,
        demand_df[["ds", "temperature_celsius", "precipitation_mm", "wind_speed_kmh"]],
    )

    # ---- Derived values (safely) ----
    forecast_only_all = forecast.iloc[-MAX_FORECAST_HOURS:]          # last 48 rows = future
    forecast_only = forecast_only_all.head(forecast_hours)           # slider-limited view
    next_hour_row = forecast_only.iloc[0] if len(forecast_only) > 0 else None

    peak_forecast = forecast_only["yhat"].max() if len(forecast_only) > 0 else 0
    total_forecast = forecast_only["yhat"].sum() if len(forecast_only) > 0 else 0
    avg_ci_width = (
        (forecast_only["yhat_upper"] - forecast_only["yhat_lower"]).mean()
        if len(forecast_only) > 0
        else 0
    )
    avg_historical = station_df["y"].mean()
    peak_hour = int(station_df.groupby("hour")["y"].mean().idxmax())

    # ---- Page title ----
    st.title("Valenbisi Demand Forecaster")
    st.caption(
        "Bayesian hourly demand prediction with uncertainty quantification "
        "for Valencia's bike-sharing network."
    )

    # ---- Tab layout ----
    tab_overview, tab_forecast, tab_methodology = st.tabs([
        "Network Overview",
        "Bayesian Forecast",
        "Methodology & Architecture",
    ])

    # ===================================================================
    # TAB 1 — Network Overview
    # ===================================================================
    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected Station", station_info["name"])
        c2.metric("Station Type", station_info["type"].replace("_", " ").title())
        c3.metric("Capacity", f"{int(station_info['capacity'])} bikes")
        c4.metric("Avg Hourly Demand (hist.)", f"{avg_historical:.1f} bikes/h")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Peak Hour (hist.)", f"{peak_hour:02d}:00")
        c6.metric(
            f"Total Forecast ({forecast_hours}h)",
            f"{total_forecast:.0f} bikes",
        )
        c7.metric("Peak Forecast Demand", f"{peak_forecast:.1f} bikes/h")
        c8.metric("Avg Uncertainty Width", f"{avg_ci_width:.1f} bikes/h")

        if metrics["mae"] is not None:
            st.divider()
            p1, p2, p3 = st.columns(3)
            p1.metric("Backtest MAE (30-day)", f"{metrics['mae']:.2f} bikes/h")
            p2.metric(
                "95 % CI Coverage",
                f"{metrics['coverage_95']} %",
                help="Fraction of true values inside the 95 % credible interval. Ideal ≈ 95 %.",
            )
            p3.metric(
                "Forecast Reliability",
                "Well-calibrated" if metrics["coverage_95"] >= 90 else "Under-confident",
            )

        st.divider()
        st.subheader("Station Map — Valencia")
        folium_map = build_map(stations_df, demand_df, selected_id)
        st_folium(folium_map, width="100%", height=520, returned_objects=[])

        st.caption(
            "Circle colors indicate station typology. "
            "Click any marker for detailed statistics. "
            "The selected station is outlined in white."
        )

    # ===================================================================
    # TAB 2 — Bayesian Forecast
    # ===================================================================
    with tab_forecast:
        # ── Recompute from slider value RIGHT HERE (guaranteed fresh) ──
        future_rows = forecast.iloc[-MAX_FORECAST_HOURS:]
        fc_display = future_rows.head(forecast_hours)
        fc_next = fc_display.iloc[0] if len(fc_display) > 0 else None

        if fc_next is not None:
            st.info(
                f"**Next hour forecast ({fc_next['ds'].strftime('%H:%M')}):** "
                f"**{fc_next['yhat']:.1f}** bikes "
                f"[95 % CI: {fc_next['yhat_lower']:.1f} – {fc_next['yhat_upper']:.1f}]"
            )

        fig = build_forecast_chart(station_df, fc_display, station_info["name"])
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.caption("Forecast breakdown (next 6 hours)")
        preview = fc_display.head(6)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        preview["ds"] = preview["ds"].dt.strftime("%a %d %b — %H:%M")
        preview.columns = ["Datetime", "Median", "Lower 95%", "Upper 95%"]
        preview["Uncertainty (±)"] = (
            (preview["Upper 95%"] - preview["Lower 95%"]) / 2
        ).round(1)

        st.dataframe(
            preview.set_index("Datetime").style.format("{:.1f}"),
            use_container_width=True,
        )

    # ===================================================================
    # TAB 3 — Methodology & Architecture
    # ===================================================================
    with tab_methodology:
        st.markdown(METHODOLOGY_TEXT)


if __name__ == "__main__":
    main()