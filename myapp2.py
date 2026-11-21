import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Portfolio + Signal Lab",
    layout="wide"
)

st.title("📈 Personal Trading Analytics App")
st.subheader("Portfolio Mode + Signal Lab")

# ====================
# 1. USER INPUT (MULTI-TICKER)
# ====================
st.sidebar.header("Portfolio Settings")

tickers = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value="AAPL,MSFT,NVDA"
)

ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.Timestamp("2023-01-01")
)

# ====================
# 2. DOWNLOAD DATA
# ====================
df = yf.download(
    ticker_list,
    start=start_date,
    auto_adjust=True
)["Close"]

if df.empty:
    st.error("No data found")
    st.stop()

# Flatten single-ticker edge case
if isinstance(df, pd.Series):
    df = df.to_frame()

# ====================
# 3. PORTFOLIO WEIGHTS
# ====================
weights = np.array([1 / len(ticker_list)] * len(ticker_list))

allocation = pd.DataFrame({
    "Ticker": ticker_list,
    "Weight": weights
})

# ====================
# 4. RETURNS
# ====================
returns = df.pct_change().dropna()

# ====================
# 5. PORTFOLIO RETURNS
# ====================
portfolio_returns = returns.dot(weights)

# ====================
# 6. CUMULATIVE PERFORMANCE
# ====================
cumulative = (1 + portfolio_returns).cumprod()

# ====================
# 7. METRICS
# ====================
total_return = (cumulative.iloc[-1] - 1) * 100

volatility = portfolio_returns.std() * np.sqrt(252) * 100

sharpe = (
    portfolio_returns.mean() /
    portfolio_returns.std()
) * np.sqrt(252)

# ====================
# 8. DISPLAY METRICS
# ====================
col1, col2, col3 = st.columns(3)

col1.metric("Portfolio Return", f"{total_return:.2f}%")
col2.metric("Volatility", f"{volatility:.2f}%")
col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

# ====================
# 9. VISUALIZATION
# ====================

# --- Allocation Pie ---
st.header("Portfolio Allocation")

fig_pie = px.pie(
    allocation,
    names="Ticker",
    values="Weight",
    title="Asset Allocation"
)

st.plotly_chart(fig_pie, use_container_width=True)

# --- Portfolio Performance ---
st.header("Portfolio Performance")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=cumulative.index,
    y=cumulative,
    name="Portfolio"
))

st.plotly_chart(fig, use_container_width=True)

# --- Drawdown ---
st.header("Drawdown Analysis")

drawdown = cumulative / cumulative.cummax() - 1

fig_dd = px.area(
    x=cumulative.index,
    y=drawdown,
    labels={"y": "Drawdown"}
)

st.plotly_chart(fig_dd, use_container_width=True)

# --- Returns Distribution ---
st.header("Returns Distribution")

fig_hist = px.histogram(
    portfolio_returns,
    nbins=50,
    title="Portfolio Daily Returns"
)

st.plotly_chart(fig_hist, use_container_width=True)

# ============
# SIGNAL LAYER
# ============
st.header("Signal Lab (Simplified)")

ma_short = portfolio_returns.rolling(5).mean()
ma_long = portfolio_returns.rolling(20).mean()

signal = np.where(ma_short > ma_long, 1, 0)

st.metric(
    "Portfolio Signal",
    "BUY" if signal[-1] == 1 else "SELL"
)

st.dataframe(
    pd.DataFrame({
        "Portfolio Return": portfolio_returns,
        "Signal": signal
    }).tail(20)
)
