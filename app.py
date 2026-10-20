# ─────────────────────────────────────────────
# AI Stock Market Analysis & Prediction Platform
# app.py — Dashboard v2.1 (deployment-ready)
# ─────────────────────────────────────────────

import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import shap
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# ─── Path helper (works locally and on Streamlit Cloud) ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(relative_path):
    return os.path.join(BASE_DIR, relative_path)


# ─── Page Configuration ───────────────────────
st.set_page_config(
    page_title="AI Stock Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stApp { background-color: #0D1117; }

    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }

    .main-title {
        font-size: 28px; font-weight: 700; color: #58A6FF;
        margin-bottom: 2px; letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 13px; color: #8B949E; margin-bottom: 0px;
    }
    .section-header {
        font-size: 14px; font-weight: 600; color: #E6EDF3;
        padding-left: 10px; border-left: 3px solid #58A6FF;
        margin-top: 8px; margin-bottom: 8px; line-height: 1.4;
    }

    [data-testid="metric-container"] {
        background-color: #161B22; border: 1px solid #30363D;
        border-radius: 8px; padding: 12px 16px;
    }
    [data-testid="metric-container"] label {
        color: #8B949E !important; font-size: 12px !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #E6EDF3 !important; font-size: 22px !important;
        font-weight: 600 !important;
    }

    .buy-signal {
        background: linear-gradient(135deg, #1a4731, #0d2b1d);
        color: #3FB950; padding: 16px 24px; border-radius: 10px;
        font-size: 26px; font-weight: 700; text-align: center;
        border: 1px solid #3FB950; letter-spacing: 2px;
    }
    .sell-signal {
        background: linear-gradient(135deg, #4d1919, #2d0f0f);
        color: #F85149; padding: 16px 24px; border-radius: 10px;
        font-size: 26px; font-weight: 700; text-align: center;
        border: 1px solid #F85149; letter-spacing: 2px;
    }
    .hold-signal {
        background: linear-gradient(135deg, #3d2e00, #261d00);
        color: #D29922; padding: 16px 24px; border-radius: 10px;
        font-size: 26px; font-weight: 700; text-align: center;
        border: 1px solid #D29922; letter-spacing: 2px;
    }

    .reason-positive {
        color: #3FB950; font-size: 12px; padding: 3px 0px;
        border-bottom: 1px solid #21262D;
    }
    .reason-negative {
        color: #F85149; font-size: 12px; padding: 3px 0px;
        border-bottom: 1px solid #21262D;
    }

    .control-panel-header {
        font-size: 16px; font-weight: 700; color: #58A6FF;
        letter-spacing: 0.5px; margin-bottom: 4px;
    }

    .about-card {
        background: #161B22; border: 1px solid #30363D;
        border-radius: 10px; padding: 14px; margin-top: 8px;
    }
    .about-row {
        font-size: 11px; color: #8B949E; padding: 3px 0;
        border-bottom: 1px solid #21262D;
    }
    .about-val { color: #E6EDF3; font-weight: 500; }

    .live-badge {
        background: #1a4731; color: #3FB950; font-size: 11px;
        font-weight: 600; padding: 3px 10px; border-radius: 20px;
        border: 1px solid #3FB950; display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ─── ARIMA Forecast ───────────────────────────
def get_arima_forecast(df, steps=7):
    try:
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()

        if len(close) < 100:
            return None, None, None

        # Use last 365 days for better trend
        train = close.tail(365).reset_index(drop=True)

        model = ARIMA(train, order=(3, 1, 1))
        fitted = model.fit()

        forecast_result = fitted.get_forecast(steps=steps)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.2)

        last_date = close.index[-1]
        future_dates = pd.bdate_range(
            start=last_date, periods=steps + 1)[1:]

        forecast_series = pd.Series(
            np.asarray(forecast_mean), index=future_dates)
        upper_series = pd.Series(
            np.asarray(conf_int.iloc[:, 1]), index=future_dates)
        lower_series = pd.Series(
            np.asarray(conf_int.iloc[:, 0]), index=future_dates)

        return forecast_series, upper_series, lower_series

    except Exception:
        return None, None, None


# ─── 30-Day Trend Forecast ────────────────────
def get_30day_forecast(df):
    try:
        from sklearn.linear_model import LinearRegression

        close = pd.to_numeric(df["Close"], errors="coerce").dropna()

        if len(close) < 90:
            return None, None, None, None, None, None

        recent = close.tail(90).values
        n = len(recent)

        X = np.arange(n).reshape(-1, 1)
        y = recent

        reg = LinearRegression()
        reg.fit(X, y)

        X_future = np.arange(n, n + 30).reshape(-1, 1)
        trend = reg.predict(X_future)

        daily_vol = float(
            pd.Series(recent).pct_change().std() * recent[-1])

        ci_width = np.linspace(daily_vol * 1.0, daily_vol * 3.5, 30)

        last_date = close.index[-1]
        future_dates = pd.bdate_range(
            start=last_date, periods=31)[1:]

        forecast = pd.Series(trend, index=future_dates)
        upper = pd.Series(trend + ci_width, index=future_dates)
        lower = pd.Series(trend - ci_width, index=future_dates)

        slope = reg.coef_[0]
        r2 = reg.score(X, y)
        daily_chg = float(slope)
        total_chg = float(trend[-1] - recent[-1])
        total_pct = total_chg / recent[-1] * 100

        return forecast, upper, lower, daily_chg, total_pct, r2

    except Exception:
        return None, None, None, None, None, None


# ─── Next-Day Trend Estimate (linear extrapolation) ───
def get_next_day_estimate(df):
    try:
        close = df["Close"].dropna().values.reshape(-1, 1)

        if len(close) < 61:
            return None, None

        last_60 = close[-60:].flatten()
        slope = (last_60[-1] - last_60[0]) / 60
        next_price = float(last_60[-1] + slope)

        direction = ("UP 📈" if next_price > last_60[-1]
                     else "DOWN 📉")

        return round(next_price, 2), direction

    except Exception:
        return None, None


# ─── Feature Columns ──────────────────────────
FEATURE_COLS = [
    "MA50", "MA200", "Golden_Cross",
    "RSI", "RSI_Overbought", "RSI_Oversold",
    "MACD", "MACD_Signal", "MACD_Hist",
    "MACD_Positive",
    "BB_Upper", "BB_Mid", "BB_Lower",
    "BB_Width", "BB_Position",
    "Volume_MA", "Volume_Ratio", "Volume_Spike",
    "Daily_Return",
    "Return_1d", "Return_3d",
    "Return_5d", "Return_10d",
    "Volatility_5d", "Volatility_10d",
    "Price_vs_MA50", "Price_vs_MA200"
]


# ─── Load Models ──────────────────────────────
@st.cache_resource
def load_models():
    return {
        "TVS Motor": joblib.load(
            get_path("models/rf_tvs_motor.pkl")),
        "Solar Industries": joblib.load(
            get_path("models/rf_solar_industries.pkl")),
        "HDFC Bank": joblib.load(
            get_path("models/rf_hdfc_bank.pkl")),
    }


# ─── Live Data Fetch ──────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_one(ticker):
    # Raising on empty data means failures are NOT cached
    df = yf.download(
        ticker,
        start="2019-01-01",
        auto_adjust=True,
        progress=False
    )
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def load_live_data():
    tickers = {
        "TVS Motor": "TVSMOTOR.NS",
        "Solar Industries": "SOLARINDS.NS",
        "HDFC Bank": "HDFCBANK.NS"
    }
    data = {}
    for name, ticker in tickers.items():
        try:
            df = fetch_one(ticker).copy()
            df["Stock"] = name
            data[name] = df
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")
    return data


# ─── Compute Indicators ───────────────────────
def compute_indicators(df):
    df = df.copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["Golden_Cross"] = (df["MA50"] > df["MA200"]).astype(int)

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    RS = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + RS))
    df["RSI_Overbought"] = (df["RSI"] > 70).astype(int)
    df["RSI_Oversold"] = (df["RSI"] < 30).astype(int)

    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    df["MACD_Positive"] = (df["MACD_Hist"] > 0).astype(int)

    # Bollinger Bands
    df["BB_Mid"] = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + (2 * std)
    df["BB_Lower"] = df["BB_Mid"] - (2 * std)
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
    df["BB_Position"] = np.where(
        df["BB_Width"] != 0,
        (df["Close"] - df["BB_Lower"]) / df["BB_Width"], 0.5)

    # Volume
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df["Volume_MA"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"]
    df["Volume_Spike"] = (df["Volume_Ratio"] > 1.5).astype(int)

    # Returns
    df["Daily_Return"] = df["Close"].pct_change() * 100
    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_3d"] = df["Close"].pct_change(3)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)
    df["Volatility_5d"] = df["Return_1d"].rolling(5).std()
    df["Volatility_10d"] = df["Return_1d"].rolling(10).std()
    df["Price_vs_MA50"] = np.where(
        df["MA50"] != 0,
        (df["Close"] - df["MA50"]) / df["MA50"], 0)
    df["Price_vs_MA200"] = np.where(
        df["MA200"] != 0,
        (df["Close"] - df["MA200"]) / df["MA200"], 0)
    df["Cumulative_Return"] = (
        1 + df["Daily_Return"] / 100).cumprod() - 1

    return df


# ─── Prediction ───────────────────────────────
def get_prediction(model, df):
    data = df[FEATURE_COLS].copy().dropna()
    if len(data) == 0:
        return "HOLD", 50.0, 50.0
    last_row = data.iloc[[-1]]
    pred_proba = model.predict_proba(last_row)[0]
    pred_class = model.predict(last_row)[0]
    confidence = pred_proba[pred_class] * 100
    up_prob = pred_proba[1] * 100
    if up_prob > 55:
        signal = "BUY"
    elif up_prob < 45:
        signal = "SELL"
    else:
        signal = "HOLD"
    return signal, confidence, up_prob


# ─── SHAP ─────────────────────────────────────
def get_shap_explanation(model, df):
    data = df[FEATURE_COLS].copy().dropna()
    if len(data) == 0:
        return None
    last_row = data.iloc[[-1]].values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(last_row)

    # Handle different SHAP output shapes across versions
    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[1])[0]
    else:
        sv = np.asarray(shap_values)
        sv = sv[0, :, 1] if sv.ndim == 3 else sv[0]

    return pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Value": last_row[0],
        "SHAP": sv
    }).sort_values("SHAP", ascending=False)


# ─── Scoreboard ───────────────────────────────
def get_scoreboard(df):
    latest = df.iloc[-1]
    scores = []

    rsi = float(latest.get("RSI", 50))
    if rsi < 30:
        scores.append(("RSI", f"{rsi:.1f}", "🟢 Bullish",
                       "Oversold — reversal likely"))
    elif rsi > 70:
        scores.append(("RSI", f"{rsi:.1f}", "🔴 Bearish",
                       "Overbought — pullback likely"))
    else:
        scores.append(("RSI", f"{rsi:.1f}", "🟡 Neutral",
                       "Normal range"))

    mh = float(latest.get("MACD_Hist", 0))
    scores.append(("MACD Histogram", f"{mh:.3f}",
                   "🟢 Bullish" if mh > 0 else "🔴 Bearish",
                   ("Positive — upward momentum" if mh > 0
                    else "Negative — downward momentum")))

    gc = int(latest.get("Golden_Cross", 0))
    scores.append(("MA Crossover",
                   "Golden ✨" if gc == 1 else "Death ☠️",
                   "🟢 Bullish" if gc == 1 else "🔴 Bearish",
                   ("MA50 > MA200 — uptrend" if gc == 1
                    else "MA50 < MA200 — downtrend")))

    bb = float(latest.get("BB_Position", 0.5))
    if bb > 0.8:
        scores.append(("Bollinger", f"{bb:.2f}", "🔴 Bearish",
                       "Near upper band"))
    elif bb < 0.2:
        scores.append(("Bollinger", f"{bb:.2f}", "🟢 Bullish",
                       "Near lower band"))
    else:
        scores.append(("Bollinger", f"{bb:.2f}", "🟡 Neutral",
                       "Mid band"))

    vr = float(latest.get("Volume_Ratio", 1))
    scores.append(("Volume", f"{vr:.2f}x",
                   "🟢 Bullish" if vr > 1.2
                   else "🔴 Bearish" if vr < 0.8
                   else "🟡 Neutral",
                   ("High conviction" if vr > 1.2
                    else "Low conviction" if vr < 0.8
                    else "Average volume")))

    return scores


# ─── Candlestick Chart ────────────────────────
PERIOD_MAP = {
    "1W": 5, "1M": 21, "3M": 63, "6M": 126,
    "1Y": 252, "2Y": 504, "All": None
}


def plot_candlestick(df, name, show_bb=False,
                     show_volume=True, period="1Y"):
    days = PERIOD_MAP.get(period)
    df_plot = (df.tail(days) if days else df).copy()

    rows = 3 if show_volume else 2
    row_heights = ([0.65, 0.2, 0.15] if show_volume
                   else [0.75, 0.25])

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.02, row_heights=row_heights)

    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot["Open"], high=df_plot["High"],
        low=df_plot["Low"], close=df_plot["Close"],
        name=name,
        increasing=dict(line=dict(color="#26A69A", width=1),
                        fillcolor="#26A69A"),
        decreasing=dict(line=dict(color="#EF5350", width=1),
                        fillcolor="#EF5350"),
        whiskerwidth=0.3
    ), row=1, col=1)

    if "MA50" in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["MA50"], name="MA50",
            line=dict(color="#2196F3", width=1.8)),
            row=1, col=1)

    if "MA200" in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["MA200"], name="MA200",
            line=dict(color="#FF9800", width=1.8)),
            row=1, col=1)

    if show_bb and "BB_Upper" in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["BB_Upper"],
            name="BB Upper",
            line=dict(color="#9C27B0", width=1, dash="dot"),
            opacity=0.8), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["BB_Lower"],
            name="BB Lower",
            line=dict(color="#9C27B0", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(156,39,176,0.06)",
            opacity=0.8), row=1, col=1)

    if "RSI" in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["RSI"], name="RSI",
            line=dict(color="#E91E63", width=1.5)),
            row=2, col=1)
        for level, color in [(70, "#EF5350"), (30, "#26A69A")]:
            fig.add_hline(
                y=level, line_dash="dash", line_color=color,
                line_width=1, opacity=0.6, row=2, col=1)

    if show_volume and "Volume" in df_plot.columns:
        vol_colors = [
            "#26A69A" if c >= o else "#EF5350"
            for c, o in zip(df_plot["Close"], df_plot["Open"])
        ]
        fig.add_trace(go.Bar(
            x=df_plot.index, y=df_plot["Volume"],
            name="Volume", marker_color=vol_colors,
            opacity=0.7, showlegend=False),
            row=3, col=1)

    fig.update_layout(
        height=700,
        paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
        font=dict(color="#C9D1D9", size=11),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="top", y=-0.05,
            xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#8B949E")),
        margin=dict(l=10, r=10, t=10, b=60),
        hovermode="x unified",
        modebar=dict(
            bgcolor="rgba(0,0,0,0)", color="#8B949E",
            activecolor="#58A6FF",
            remove=["lasso2d", "select2d", "autoScale2d"])
    )

    for i in range(1, rows + 1):
        fig.update_xaxes(
            showgrid=True, gridcolor="#21262D", gridwidth=0.5,
            showline=True, linecolor="#30363D", row=i, col=1)
        fig.update_yaxes(
            showgrid=True, gridcolor="#21262D", gridwidth=0.5,
            showline=True, linecolor="#30363D", row=i, col=1)

    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100],
                     row=2, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=3, col=1)

    return fig


# ─── MACD Chart ───────────────────────────────
def plot_macd(df, period="1Y"):
    days = PERIOD_MAP.get(period)
    df_plot = (df.tail(days) if days else df).copy()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.45], vertical_spacing=0.03)

    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot["Close"], name="Close Price",
        line=dict(color="#58A6FF", width=1.8)),
        row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot["MACD"], name="MACD",
        line=dict(color="#2196F3", width=1.5)),
        row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot["MACD_Signal"], name="Signal",
        line=dict(color="#FF9800", width=1.5)),
        row=2, col=1)

    hist_colors = ["#26A69A" if v >= 0 else "#EF5350"
                   for v in df_plot["MACD_Hist"]]
    fig.add_trace(go.Bar(
        x=df_plot.index, y=df_plot["MACD_Hist"],
        name="Histogram", marker_color=hist_colors, opacity=0.8),
        row=2, col=1)

    fig.update_layout(
        height=500,
        paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
        font=dict(color="#C9D1D9", size=11),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=10, b=40),
        hovermode="x unified",
        legend=dict(
            orientation="h", y=-0.08, bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#8B949E")),
        modebar=dict(
            bgcolor="rgba(0,0,0,0)", color="#8B949E",
            activecolor="#58A6FF",
            remove=["lasso2d", "select2d"])
    )

    for i in [1, 2]:
        fig.update_xaxes(showgrid=True, gridcolor="#21262D",
                         row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#21262D",
                         row=i, col=1)

    return fig


# ══════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════

# ─── Header ───────────────────────────────────
col_title, col_badge = st.columns([6, 1])
with col_title:
    st.markdown(
        '<p class="main-title">'
        '📈 AI Stock Intelligence Platform</p>',
        unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">'
        'AI-powered BUY / HOLD / SELL · '
        'Indian NSE Stocks · '
        'Explainable AI · '
        'RCOEM Final Year Project 2026</p>',
        unsafe_allow_html=True)
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<span class="live-badge">🟢 LIVE DATA</span>',
        unsafe_allow_html=True)

st.divider()

# ─── Load ─────────────────────────────────────
with st.spinner("Fetching latest market data..."):
    all_models = load_models()
    all_data = load_live_data()

if len(all_data) < 3:
    st.error(
        "Yahoo Finance didn't return data for all stocks "
        "(usually a temporary rate limit). "
        "Please wait a minute and click Refresh.")
    if st.button("🔄 Retry"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

processed = {n: compute_indicators(d)
             for n, d in all_data.items()}

# ─── Sidebar — Control Panel ──────────────────
with st.sidebar:

    st.markdown(
        '<p class="control-panel-header">🎛️ Control Panel</p>',
        unsafe_allow_html=True)
    st.markdown(
        "<small style='color:#8B949E'>"
        "Select stock and chart options</small>",
        unsafe_allow_html=True)
    st.divider()

    st.markdown(
        "<small style='color:#58A6FF;font-weight:600'>"
        "SECTOR</small>", unsafe_allow_html=True)
    sector = st.selectbox(
        "sector",
        ["Automobile & EV",
         "Defence & Industrials",
         "Banking & Finance"],
        label_visibility="collapsed")

    sector_map = {
        "Automobile & EV": "TVS Motor",
        "Defence & Industrials": "Solar Industries",
        "Banking & Finance": "HDFC Bank"
    }
    selected_stock = sector_map[sector]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<small style='color:#58A6FF;font-weight:600'>"
        "STOCK</small>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='background:#1F2937;"
        f"border:1px solid #30363D;border-radius:8px;"
        f"padding:10px 14px;color:#58A6FF;"
        f"font-weight:600;font-size:14px'>"
        f"📌 {selected_stock}</div>",
        unsafe_allow_html=True)

    st.divider()

    st.markdown(
        "<small style='color:#58A6FF;font-weight:600'>"
        "CHART SETTINGS</small>", unsafe_allow_html=True)

    period = st.select_slider(
        "Time period",
        options=["1W", "1M", "3M", "6M", "1Y", "2Y", "All"],
        value="1Y")

    show_bb = st.toggle("Bollinger Bands", value=False)
    show_vol = st.toggle("Volume bars", value=True)

    st.divider()

    st.markdown(
        "<small style='color:#58A6FF;font-weight:600'>"
        "DATA REFRESH</small>", unsafe_allow_html=True)

    IST = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(IST)
    st.markdown(
        f"<small style='color:#8B949E'>"
        f"Last: {now_ist.strftime('%H:%M:%S')} IST</small>",
        unsafe_allow_html=True)

    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    is_open = (
        now_ist.weekday() < 5
        and 9 * 60 + 15
        <= now_ist.hour * 60 + now_ist.minute
        <= 15 * 60 + 30
    )
    if is_open:
        st.markdown(
            "<small style='color:#3FB950'>"
            "🟢 Market OPEN — data cached for 5 min"
            "</small>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<small style='color:#F85149'>"
            "🔴 Market CLOSED — showing last price"
            "</small>", unsafe_allow_html=True)

    st.divider()

    st.markdown(
        '<p class="control-panel-header">ℹ️ About</p>',
        unsafe_allow_html=True)

    st.markdown("""
<div class="about-card">

<div class="about-row">
🏫 <span class="about-val">RCOEM, Nagpur</span>
</div>

<div class="about-row">
📚 <span class="about-val">B.Tech ECS — Sem VI</span>
</div>

<div class="about-row">
👥 <span class="about-val">
Rishikesh · Saie · Amogh · Abhishek</span>
</div>

<div class="about-row">
🧑‍🏫 <span class="about-val">Guide: Dr. Jayu Kalambe</span>
</div>

<div class="about-row" style="margin-top:8px">
📊 <span class="about-val">Data: 6+ yrs · 5,456 rows</span>
</div>

<div class="about-row">
🤖 <span class="about-val">Model: Gradient Boosting</span>
</div>

<div class="about-row">
🔍 <span class="about-val">XAI: SHAP explainability</span>
</div>

<div class="about-row">
📈 <span class="about-val">Features: 27 indicators</span>
</div>

<div class="about-row" style="margin-top:8px">
✅ TVS Motor: 53.5% acc
</div>
<div class="about-row">
✅ Solar Ind: 60.9% acc
</div>
<div class="about-row">
✅ HDFC Bank: 56.7% acc
</div>

<div class="about-row" style="margin-top:8px">
🐙 <span class="about-val">GitHub: ai-stock-platform</span>
</div>

</div>
""", unsafe_allow_html=True)

# ─── Selected Stock ───────────────────────────
df = processed[selected_stock]
model = all_models[selected_stock]

if df is None or len(df) < 2:
    st.error("Unable to load market data. "
             "Please click Refresh to try again.")
    st.stop()

df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna(subset=["Close"])

if len(df) < 2:
    st.error("Insufficient data loaded. "
             "Please click Refresh.")
    st.stop()

# ─── Top Metrics ──────────────────────────────
st.markdown(
    f'<p class="section-header">'
    f'📌 {selected_stock} — Live Overview</p>',
    unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)

latest = float(df["Close"].iloc[-1])
prev = float(df["Close"].iloc[-2])
chg = latest - prev
chg_pct = (chg / prev) * 100
h52 = float(df["Close"].tail(252).max())
l52 = float(df["Close"].tail(252).min())
avol = float(df["Volume"].tail(20).mean())
tot_ret = float(
    (df["Close"].iloc[-1] - df["Close"].iloc[0])
    / df["Close"].iloc[0] * 100)
rsi_now = (float(df["RSI"].dropna().iloc[-1])
           if "RSI" in df.columns
           and len(df["RSI"].dropna()) > 0
           else 50.0)

c1.metric("Current Price", f"₹{latest:,.2f}",
          f"{chg_pct:+.2f}%")
c2.metric("Prev Close", f"₹{prev:,.2f}")
c3.metric("52W High", f"₹{h52:,.2f}")
c4.metric("52W Low", f"₹{l52:,.2f}")
c5.metric("Avg Vol (20d)", f"{avol/1e6:.2f}M")
c6.metric("Total Return", f"{tot_ret:.1f}%")

st.divider()

# ─── Forecasts (computed once, used in both columns) ───
with st.spinner("Computing forecasts..."):
    # Short term — ARIMA (7 trading days)
    arima_f, arima_u, arima_l = get_arima_forecast(df, steps=7)

    # Long term — Linear Regression (30 trading days)
    lr_f = lr_u = lr_l = None
    lr_daily = lr_pct = lr_r2 = None
    lr_result = get_30day_forecast(df)
    if lr_result[0] is not None:
        (lr_f, lr_u, lr_l,
         lr_daily, lr_pct, lr_r2) = lr_result

# ─── Main Area ────────────────────────────────
left_col, right_col = st.columns([3, 1])

with left_col:

    st.markdown(
        '<p class="section-header">'
        f'🕯️ Price Chart — {selected_stock}</p>',
        unsafe_allow_html=True)

    chart_tab, macd_tab, ret_tab = st.tabs([
        "📊 Candlestick + RSI",
        "📉 MACD",
        "📈 Returns"
    ])

    with chart_tab:

        fig = plot_candlestick(
            df, selected_stock,
            show_bb=show_bb,
            show_volume=show_vol,
            period=period)

        if arima_f is not None:
            fig.add_trace(go.Scatter(
                x=arima_f.index, y=arima_f.values,
                name="ARIMA 7d Forecast",
                line=dict(color="#FF6B6B", width=2,
                          dash="dash"),
                opacity=1.0), row=1, col=1)

        if lr_f is not None:
            fig.add_trace(go.Scatter(
                x=lr_u.index, y=lr_u.values,
                name="30d Upper Band",
                line=dict(color="rgba(0,0,0,0)", width=0),
                showlegend=False), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=lr_l.index, y=lr_l.values,
                name="30d Confidence",
                line=dict(color="rgba(0,0,0,0)", width=0),
                fill="tonexty",
                fillcolor="rgba(33,150,243,0.10)",
                showlegend=True), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=lr_f.index, y=lr_f.values,
                name="30d Trend Forecast",
                line=dict(color="#2196F3", width=2,
                          dash="dot"),
                opacity=1.0), row=1, col=1)

        # Today vertical line
        last_date = str(df["Close"].dropna().index[-1].date())
        fig.add_vline(
            x=last_date, line_dash="dot",
            line_color="#8B949E", line_width=1.5)

        fig.add_annotation(
            x=last_date,
            y=float(df["Close"].iloc[-1]),
            text="Today", showarrow=True,
            arrowhead=1, arrowcolor="#8B949E",
            font=dict(color="#8B949E", size=11))

        st.plotly_chart(
            fig, use_container_width=True,
            config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": [
                    "lasso2d", "select2d", "autoScale2d"],
                "displaylogo": False,
                "scrollZoom": True
            })

        st.markdown(
            "<small style='color:#8B949E'>"
            "🔴 Red dashed = ARIMA 7-day · "
            "🔵 Blue dotted = 30-day trend · "
            "Blue shaded = confidence interval"
            "</small>", unsafe_allow_html=True)

        col_f1, col_f2, col_f3, col_f4 = st.columns(4)

        last_p = float(df["Close"].iloc[-1])

        col_f1.metric("Current Price", f"₹{last_p:,.2f}")

        if arima_f is not None:
            arima_7_p = float(arima_f.iloc[-1])
            arima_chg = (arima_7_p - last_p) / last_p * 100
            col_f2.metric("ARIMA 7-day",
                          f"₹{arima_7_p:,.2f}",
                          f"{arima_chg:+.1f}%")

        if lr_f is not None:
            lr_30_p = float(lr_f.iloc[-1])
            lr_chg = (lr_30_p - last_p) / last_p * 100
            col_f3.metric("30-day Trend Target",
                          f"₹{lr_30_p:,.2f}",
                          f"{lr_chg:+.1f}%")
            col_f4.metric("Trend R² Score",
                          f"{lr_r2:.3f}",
                          "Strong" if lr_r2 > 0.85
                          else "Moderate")

    with macd_tab:
        fig_macd = plot_macd(df, period)
        st.plotly_chart(
            fig_macd, use_container_width=True,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "scrollZoom": True
            })

    with ret_tab:
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Scatter(
            x=df.index,
            y=df["Cumulative_Return"] * 100,
            name="Cumulative Return",
            line=dict(color="#58A6FF", width=2),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.08)"))
        fig_ret.add_hline(
            y=0, line_color="#8B949E",
            line_dash="dash", line_width=1)
        fig_ret.update_layout(
            height=500,
            paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
            font=dict(color="#C9D1D9"),
            margin=dict(l=10, r=10, t=20, b=20),
            yaxis_title="Cumulative Return (%)",
            xaxis=dict(showgrid=True, gridcolor="#21262D"),
            yaxis=dict(showgrid=True, gridcolor="#21262D"))
        st.plotly_chart(
            fig_ret, use_container_width=True,
            config={"displaylogo": False})

    st.divider()

    # Scoreboard
    st.markdown(
        '<p class="section-header">'
        '📋 Technical Signal Scoreboard</p>',
        unsafe_allow_html=True)

    scores = get_scoreboard(df)
    score_df = pd.DataFrame(
        scores,
        columns=["Indicator", "Value", "Signal", "Explanation"])

    bull = sum(1 for s in scores if "Bullish" in s[2])
    bear = sum(1 for s in scores if "Bearish" in s[2])
    neut = sum(1 for s in scores if "Neutral" in s[2])

    sb1, sb2, sb3 = st.columns(3)
    sb1.metric("🟢 Bullish signals", bull)
    sb2.metric("🟡 Neutral signals", neut)
    sb3.metric("🔴 Bearish signals", bear)

    st.dataframe(
        score_df,
        use_container_width=True,
        hide_index=True,
        height=215,
        column_config={
            "Indicator": st.column_config.TextColumn(
                width="medium"),
            "Signal": st.column_config.TextColumn(
                width="medium"),
            "Explanation": st.column_config.TextColumn(
                width="large"),
        })

with right_col:

    # Recommendation
    st.markdown(
        '<p class="section-header">🎯 ML Recommendation</p>',
        unsafe_allow_html=True)

    signal, confidence, up_prob = get_prediction(model, df)

    signal_class = {
        "BUY": "buy-signal",
        "SELL": "sell-signal",
        "HOLD": "hold-signal"
    }.get(signal, "hold-signal")

    emoji = {
        "BUY": "⬆️", "SELL": "⬇️", "HOLD": "➡️"
    }.get(signal, "➡️")

    st.markdown(
        f'<div class="{signal_class}">{emoji} {signal}</div>',
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    ca, cb = st.columns(2)
    ca.metric("Confidence", f"{confidence:.1f}%")
    cb.metric("UP Probability", f"{up_prob:.1f}%")

    st.progress(min(int(confidence), 100))

    st.divider()

    # Forecast Cards
    st.markdown(
        '<p class="section-header">📉 Price Forecasts</p>',
        unsafe_allow_html=True)

    last_p = float(df["Close"].iloc[-1])

    if arima_f is not None:
        arima_7_p = float(arima_f.iloc[-1])
        arima_chg = (arima_7_p - last_p) / last_p * 100
        st.markdown(
            "<small style='color:#FF6B6B;font-weight:600'>"
            "ARIMA — 7 DAY</small>",
            unsafe_allow_html=True)
        ca1, ca2 = st.columns(2)
        ca1.metric("Target", f"₹{arima_7_p:,.2f}",
                   f"{arima_chg:+.1f}%")
        ca2.metric("Model", "ARIMA(3,1,1)")

    st.markdown("<br>", unsafe_allow_html=True)

    if lr_f is not None:
        lr_7p = float(lr_f.iloc[6])    # 7th trading day
        lr_30p = float(lr_f.iloc[-1])  # 30th trading day
        lr_7c = (lr_7p - last_p) / last_p * 100
        lr_30c = (lr_30p - last_p) / last_p * 100

        st.markdown(
            "<small style='color:#2196F3;font-weight:600'>"
            "TREND REGRESSION — 30 DAY</small>",
            unsafe_allow_html=True)

        cb1, cb2 = st.columns(2)
        cb1.metric("7-day", f"₹{lr_7p:,.2f}", f"{lr_7c:+.1f}%")
        cb2.metric("30-day", f"₹{lr_30p:,.2f}",
                   f"{lr_30c:+.1f}%")

        fig_spark = go.Figure()
        fig_spark.add_trace(go.Scatter(
            y=lr_f.values, mode="lines",
            line=dict(color="#2196F3", width=2),
            fill="tozeroy",
            fillcolor="rgba(33,150,243,0.08)"))
        fig_spark.update_layout(
            height=70,
            paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False))
        st.plotly_chart(
            fig_spark, use_container_width=True,
            config={"displayModeBar": False})

        trend_dir = ("📈 Upward trend" if lr_30c > 0
                     else "📉 Downward trend")
        trend_col = "#3FB950" if lr_30c > 0 else "#F85149"

        st.markdown(
            f"<small style='color:{trend_col};"
            f"font-weight:600'>"
            f"{trend_dir} · R²: {lr_r2:.3f}</small>",
            unsafe_allow_html=True)

        st.markdown(
            "<small style='color:#8B949E'>"
            "Linear regression on last 90 trading days · "
            "Indicative only</small>",
            unsafe_allow_html=True)

    st.divider()

    # Next-Day Trend Estimate
    st.markdown(
        '<p class="section-header">'
        '📐 Next-Day Trend Estimate</p>',
        unsafe_allow_html=True)

    nd_price, nd_dir = get_next_day_estimate(df)

    if nd_price is not None:
        current_p = float(df["Close"].iloc[-1])
        nd_chg = (nd_price - current_p) / current_p * 100

        lc1, lc2 = st.columns(2)
        lc1.metric("Estimated Price", f"₹{nd_price:,.2f}",
                   f"{nd_chg:+.2f}%")
        lc2.metric("Direction", nd_dir)

        st.markdown(
            "<small style='color:#8B949E'>"
            "Linear extrapolation of the last 60 trading "
            "days. Indicative only.</small>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            "<small style='color:#8B949E'>"
            "Insufficient data for estimate.</small>",
            unsafe_allow_html=True)

    st.divider()

    # RSI Gauge
    st.markdown(
        '<p class="section-header">📊 RSI Gauge</p>',
        unsafe_allow_html=True)

    rsi_color = ("#EF5350" if rsi_now > 70
                 else "#26A69A" if rsi_now < 30
                 else "#2196F3")

    rsi_label = ("OVERBOUGHT" if rsi_now > 70
                 else "OVERSOLD" if rsi_now < 30
                 else "NEUTRAL")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rsi_now,
        number={"font": {"size": 36, "color": rsi_color},
                "suffix": ""},
        title={
            "text": (f"RSI (14-day)<br>"
                     f"<span style='font-size:12px;"
                     f"color:{rsi_color}'>"
                     f"{rsi_label}</span>"),
            "font": {"size": 13, "color": "#C9D1D9"}
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickvals": [0, 20, 30, 50, 70, 80, 100],
                "ticktext": ["0", "20", "30", "50",
                             "70", "80", "100"],
                "tickcolor": "#8B949E",
                "tickfont": {"size": 10},
            },
            "bar": {"color": rsi_color, "thickness": 0.25},
            "bgcolor": "#161B22",
            "borderwidth": 1,
            "bordercolor": "#30363D",
            "steps": [
                {"range": [0, 30],
                 "color": "rgba(38,166,154,0.15)"},
                {"range": [30, 70],
                 "color": "rgba(33,150,243,0.08)"},
                {"range": [70, 100],
                 "color": "rgba(239,83,80,0.15)"}
            ],
            "threshold": {
                "line": {"color": "#E6EDF3", "width": 2},
                "thickness": 0.8,
                "value": rsi_now
            }
        },
        domain={"x": [0, 1], "y": [0, 1]}
    ))

    fig_gauge.update_layout(
        height=240,
        paper_bgcolor="#0D1117",
        font=dict(color="#C9D1D9"),
        margin=dict(l=20, r=20, t=60, b=20))

    st.plotly_chart(
        fig_gauge, use_container_width=True,
        config={"displayModeBar": False})

    st.divider()

    # SHAP Explanation
    st.markdown(
        '<p class="section-header">'
        '💡 Why This Recommendation?</p>',
        unsafe_allow_html=True)

    try:
        with st.spinner("Computing SHAP..."):
            explanation = get_shap_explanation(model, df)
    except Exception as e:
        explanation = None
        st.warning(f"SHAP explanation unavailable: {e}")

    if explanation is not None:
        top_pos = explanation[explanation["SHAP"] > 0].head(4)
        top_neg = explanation[explanation["SHAP"] < 0].tail(3)

        st.markdown(
            "<small style='color:#3FB950;font-weight:600'>"
            "✅ BULLISH FACTORS</small>",
            unsafe_allow_html=True)
        for _, row in top_pos.iterrows():
            st.markdown(
                f'<p class="reason-positive">'
                f'▲ {row["Feature"]} '
                f'· +{row["SHAP"]:.3f}</p>',
                unsafe_allow_html=True)

        st.markdown(
            "<small style='color:#F85149;font-weight:600'>"
            "⚠️ BEARISH FACTORS</small>",
            unsafe_allow_html=True)
        for _, row in top_neg.iterrows():
            st.markdown(
                f'<p class="reason-negative">'
                f'▼ {row["Feature"]} '
                f'· {row["SHAP"]:.3f}</p>',
                unsafe_allow_html=True)

st.divider()

# ─── Portfolio Row ────────────────────────────
st.markdown(
    '<p class="section-header">'
    '🏦 Portfolio Overview — All Stocks</p>',
    unsafe_allow_html=True)

port_cols = st.columns(3)
stock_colors = {
    "TVS Motor": "#2196F3",
    "Solar Industries": "#4CAF50",
    "HDFC Bank": "#FF9800"
}

for i, (sname, sdf) in enumerate(processed.items()):
    smodel = all_models[sname]

    if sdf is None or len(sdf) < 2:
        continue
    sdf = sdf.copy()
    sdf["Close"] = pd.to_numeric(sdf["Close"], errors="coerce")
    sdf = sdf.dropna(subset=["Close"])
    if len(sdf) < 2:
        continue

    sig, conf, up_p = get_prediction(smodel, sdf)
    ret = float((sdf["Close"].iloc[-1] - sdf["Close"].iloc[0])
                / sdf["Close"].iloc[0] * 100)
    price = float(sdf["Close"].iloc[-1])
    color = stock_colors[sname]

    sig_color = {
        "BUY": "#3FB950",
        "SELL": "#F85149",
        "HOLD": "#D29922"
    }.get(sig, "#8B949E")

    with port_cols[i]:
        st.markdown(
            f"<div style='background:#161B22;"
            f"border:1px solid #30363D;border-radius:10px;"
            f"padding:14px;border-top:3px solid {color}'>"
            f"<div style='font-size:14px;font-weight:700;"
            f"color:{color}'>{sname}</div>"
            f"<div style='font-size:22px;font-weight:700;"
            f"color:#E6EDF3;margin:6px 0'>₹{price:,.2f}</div>"
            f"<div style='font-size:20px;font-weight:700;"
            f"color:{sig_color}'>{sig}</div>"
            f"<div style='font-size:12px;color:#8B949E;"
            f"margin-top:4px'>"
            f"Confidence: {conf:.1f}% · "
            f"Return: {ret:.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True)

st.divider()

# ─── Footer ───────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#8B949E;
            font-size:11px;padding:8px'>
📌 For educational purposes only — not financial advice
&nbsp;·&nbsp;
Data: Yahoo Finance via yfinance
&nbsp;·&nbsp;
Data cached for 5 minutes · use Refresh for latest
&nbsp;·&nbsp;
RCOEM B.Tech Final Year Project 2026
</div>
""", unsafe_allow_html=True)
