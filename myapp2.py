"""Bước 3 — myapp2.py: Multi-Asset Equally Weighted Portfolio App"""
from datetime import date, datetime
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

st.set_page_config(page_title="Personal Trading Analytics App - Portfolio Mode", page_icon="📈", layout="wide")

st.title("📈 Personal Trading Analytics App")
st.subheader("Portfolio Mode + Signal Lab")

with st.sidebar:
    st.header("⚙️ Settings")
    tickers_input = st.text_input("Tickers", "AAPL,MSFT,NVDA")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    start_date = st.date_input("Start Date", date(2023, 1, 1))


@st.cache_data(ttl=3600)
def fetch_portfolio_data(symbols: list, start: str):
    raw = yf.download(symbols, start=start, end=str(datetime.today().date()), auto_adjust=True)
    if raw.empty:
        return pd.DataFrame()
    close_df = raw["Close"].copy() if "Close" in raw else raw.copy()
    if isinstance(close_df, pd.Series):
        close_df = close_df.to_frame(name=symbols[0])
    if isinstance(close_df.columns, pd.MultiIndex):
        close_df.columns = close_df.columns.get_level_values(-1)
    close_df.dropna(how="all", inplace=True)
    close_df.ffill(inplace=True)
    close_df.dropna(inplace=True)
    return close_df


if not tickers:
    st.warning("Vui lòng nhập ít nhất 1 mã cổ phiếu.")
else:
    close_df = fetch_portfolio_data(tickers, str(start_date))
    if close_df.empty:
        st.error("Không tải được dữ liệu cho danh sách mã này.")
    else:
        returns_df = close_df.pct_change().dropna()
        n = len(close_df.columns)
        weights = np.ones(n) / n

        port_ret = returns_df.dot(weights)
        wealth = (1 + port_ret).cumprod()
        cum_ret = wealth - 1

        total_return = wealth.iloc[-1] - 1
        vol = port_ret.std() * np.sqrt(252)
        sharpe = (port_ret.mean() * 252) / vol if vol > 0 else 0.0
        drawdown = (wealth - wealth.cummax()) / wealth.cummax()

        # 3 metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Portfolio Return", f"{total_return * 100:.2f}%")
        c2.metric("Volatility", f"{vol * 100:.2f}%")
        c3.metric("Sharpe Ratio", f"{sharpe:.2f}")

        # Portfolio Allocation: Pie chart
        st.subheader("Portfolio Allocation")
        fig_alloc = px.pie(names=list(close_df.columns), values=weights * 100)
        st.plotly_chart(fig_alloc, use_container_width=True)

        # Portfolio Performance: Line chart
        st.subheader("Portfolio Performance")
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret * 100, mode="lines", name="Portfolio"))
        fig_perf.update_layout(height=380, yaxis_title="Cumulative Return (%)", yaxis=dict(ticksuffix="%"))
        st.plotly_chart(fig_perf, use_container_width=True)

        # Drawdown Analysis: Area chart
        st.subheader("Drawdown Analysis")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown * 100, fill="tozeroy", mode="lines", name="Drawdown", line=dict(color="red")))
        fig_dd.update_layout(height=350, yaxis_title="Drawdown (%)", yaxis=dict(ticksuffix="%"))
        st.plotly_chart(fig_dd, use_container_width=True)

        # Returns Distribution: Histogram
        st.subheader("Returns Distribution")
        fig_hist = px.histogram(port_ret * 100, nbins=50, labels={"value": "Portfolio Daily Return (%)"})
        fig_hist.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Signal Lab (Simplified)
        st.subheader("Signal Lab (Simplified)")
        ma5 = wealth.rolling(5).mean()
        ma20 = wealth.rolling(20).mean()
        port_signal = pd.Series(np.where(ma5 > ma20, "BUY", "SELL"), index=port_ret.index)

        st.metric("Portfolio Signal", port_signal.iloc[-1])
        sig_df = pd.DataFrame({
            "Portfolio Return": port_ret,
            "Signal": port_signal
        })
        st.dataframe(sig_df.tail(20), use_container_width=True)
