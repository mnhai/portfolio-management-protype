# 1. Imports
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# 2. App configuration
st.set_page_config(
    page_title="Portfolio + Signal Lab",
    layout="wide"
)

st.title("📈 Personal Trading Analytics App")
st.subheader("Portfolio + Signal Lab")

# 3. Sidebar inputs (user controls)
st.sidebar.header("Settings")

ticker = st.sidebar.text_input(
    "Ticker",
    value="AAPL"
)

start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.Timestamp("2023-01-01")
)

# 4. Download stock data
df = yf.download(
    ticker,
    start=start_date,
    auto_adjust=True
)

if df.empty:
    st.error("No data found")
    st.stop()

# Flatten MultiIndex columns
# 5. Handle Yahoo Finance format issues
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

st.write(df.columns)  # should now show Close, High, Low...

# 6. Extract closing price
close = df["Close"].squeeze().dropna()

# Moving averages
# 7. Technical indicators
df["MA20"] = close.rolling(20).mean()
df["MA50"] = close.rolling(50).mean()

# Returns
df["Returns"] = close.pct_change()

# Metrics
# 8. Performance metrics
total_return = (
    close.iloc[-1] / close.iloc[0] - 1
) * 100

volatility = (
    df["Returns"].std() *
    np.sqrt(252) * 100
)

sharpe = (
    df["Returns"].mean() /
    df["Returns"].std()
) * np.sqrt(252)

# Layout
# 9. Dashboard metrics display
col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Return",
    f"{total_return:.2f}%"
)

col2.metric(
    "Annual Volatility",
    f"{volatility:.2f}%"
)

col3.metric(
    "Sharpe Ratio",
    f"{sharpe:.2f}"
)

# Price chart
st.header("Price Chart")

# 10. Price chart (interactive)
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        name="Close"
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MA20"],
        name="MA20"
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MA50"],
        name="MA50"
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# 11. Returns distribution
st.header("Returns Distribution")

hist = px.histogram(
    df.dropna(),
    x="Returns",
    nbins=50
)

st.plotly_chart(
    hist,
    use_container_width=True
)

# 12. Drawdown analysis (risk view)
st.header("Drawdown Analysis")

cum = (1 + df["Returns"]).cumprod()
drawdown = cum / cum.cummax() - 1

dd_fig = px.area(
    x=df.index,
    y=drawdown,
    labels={"y": "Drawdown"}
)

st.plotly_chart(
    dd_fig,
    use_container_width=True
)

# Signal Lab
# 13. Simple trading signal (strategy logic)
st.header("Signal Lab")

df["Signal"] = np.where(
    df["MA20"] > df["MA50"],
    1,
    0
)

signal = "BUY" if df["Signal"].iloc[-1] else "SELL"

st.metric(
    "Current Signal",
    signal
)

# 14. Data table
st.dataframe(
    df.tail(20)
)
