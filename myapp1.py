"""Bước 2 — myapp1.py: Personal Trading Analytics App"""
from datetime import date, datetime
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

st.set_page_config(page_title="Personal Trading Analytics App", page_icon="📈", layout="wide")

st.title("📈 Personal Trading Analytics App")
st.subheader("Portfolio + Signal Lab")

with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Ticker", "AAPL").strip().upper()
    start_date = st.date_input("Start Date", date(2023, 1, 1))


@st.cache_data(ttl=3600)
def load_data(symbol: str, start: str):
    df = yf.download(symbol, start=start, end=str(datetime.today().date()), auto_adjust=True, multi_level_index=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


if ticker:
    df = load_data(ticker, str(start_date))
    if df.empty:
        st.error("Không tìm thấy dữ liệu cho mã cổ phiếu này.")
    else:
        df = df.copy()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        df["Returns"] = df["Close"].pct_change()
        df["Signal"] = np.where(df["MA20"] > df["MA50"], "BUY", "SELL")

        clean_ret = df["Returns"].dropna()
        total_return = (df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1
        annual_vol = clean_ret.std() * np.sqrt(252)
        sharpe = (clean_ret.mean() * 252) / annual_vol if annual_vol > 0 else 0.0

        cum_max = df["Close"].cummax()
        drawdown = (df["Close"] - cum_max) / cum_max

        # 3 metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Return", f"{total_return * 100:.2f}%")
        c2.metric("Annual Volatility", f"{annual_vol * 100:.2f}%")
        c3.metric("Sharpe Ratio", f"{sharpe:.2f}")

        # Price Chart: Close, MA20, MA50
        st.subheader("Price Chart")
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
        fig_price.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20"))
        fig_price.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))
        fig_price.update_layout(height=420, hovermode="x unified", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_price, use_container_width=True)

        # Returns Distribution (50 bins)
        st.subheader("Returns Distribution")
        fig_hist = px.histogram(clean_ret * 100, nbins=50, labels={"value": "Daily Return (%)"})
        fig_hist.update_layout(height=350, showlegend=False, yaxis_title="Frequency")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Drawdown Analysis (Area chart)
        st.subheader("Drawdown Analysis")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown * 100, fill="tozeroy", mode="lines", name="Drawdown", line=dict(color="red")))
        fig_dd.update_layout(height=350, yaxis_title="Drawdown (%)", yaxis=dict(ticksuffix="%"))
        st.plotly_chart(fig_dd, use_container_width=True)

        # Signal Lab
        st.subheader("Signal Lab")
        current_signal = df["Signal"].iloc[-1]
        st.metric("Current Signal", current_signal)

        cols_to_show = [c for c in ["Close", "High", "Low", "Open", "Volume", "MA20", "MA50", "Returns", "Signal"] if c in df.columns]
        st.dataframe(df[cols_to_show].tail(20), use_container_width=True)
