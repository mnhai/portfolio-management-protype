"""Bước 5 — myapp7.py: Thêm Rolling Beta Engine (dựa trên myapp5)"""
from datetime import date, datetime
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

st.set_page_config(page_title="Bloomberg-Style Portfolio Terminal", page_icon="📈", layout="wide")

st.title("📈 Bloomberg-Style Portfolio Terminal")

with st.sidebar:
    st.header("⚙️ Settings")
    tickers_input = st.text_input("Tickers", "AAPL,MSFT,NVDA")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    start_date = st.date_input("Start Date", date(2023, 1, 1))

    st.subheader("Weights")
    raw_weights = {}
    default_w = 1.0 / len(tickers) if tickers else 1.0
    for t in tickers:
        raw_weights[t] = st.slider(f"{t} Weight", 0.0, 1.0, float(default_w), 0.05)

    total_w = sum(raw_weights.values())
    weights_dict = {k: v / total_w for k, v in raw_weights.items()} if total_w > 0 else {k: 1.0 / len(tickers) for k in tickers}


@st.cache_data(ttl=3600)
def fetch_data(assets: list, start: str):
    symbols = list(dict.fromkeys(assets + ["SPY"]))
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
    close_df = fetch_data(tickers, str(start_date))
    if close_df.empty or "SPY" not in close_df.columns:
        st.error("Không tải được dữ liệu danh mục hoặc SPY.")
    else:
        returns_df = close_df.pct_change().dropna()
        w_vec = np.array([weights_dict[t] for t in tickers])

        port_ret = returns_df[tickers].dot(w_vec)
        spy_ret = returns_df["SPY"]

        port_wealth = (1 + port_ret).cumprod()
        spy_wealth = (1 + spy_ret).cumprod()
        port_cum = port_wealth - 1
        spy_cum = spy_wealth - 1

        total_port_ret = port_cum.iloc[-1]
        total_spy_ret = spy_cum.iloc[-1]
        alpha_vs_spy = total_port_ret - total_spy_ret

        port_vol = port_ret.std() * np.sqrt(252)
        port_sharpe = (port_ret.mean() * 252) / port_vol if port_vol > 0 else 0.0

        port_dd = (port_wealth - port_wealth.cummax()) / port_wealth.cummax()
        spy_dd = (spy_wealth - spy_wealth.cummax()) / spy_wealth.cummax()

        roll_vol_port = port_ret.rolling(20).std() * np.sqrt(252)
        roll_vol_spy = spy_ret.rolling(20).std() * np.sqrt(252)

        # Rolling Beta (20D) & Rolling Alpha (tính trong code, chưa hiển thị)
        rolling_beta = port_ret.rolling(20).cov(spy_ret) / spy_ret.rolling(20).var()
        latest_beta = float(rolling_beta.dropna().iloc[-1]) if not rolling_beta.dropna().empty else 1.0
        rolling_alpha = (port_ret.rolling(20).mean() * 252) - rolling_beta * (spy_ret.rolling(20).mean() * 252)

        # Signals
        sig_matrix = pd.DataFrame(index=close_df.index)
        for t in tickers:
            ma5 = close_df[t].rolling(5).mean()
            ma20 = close_df[t].rolling(20).mean()
            sig_matrix[t] = np.where(ma5 > ma20, 1, 0)
        avg_signal = sig_matrix.mean(axis=1)
        port_sig_label = "BUY" if avg_signal.iloc[-1] >= 0.5 else "SELL"

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio", "⚠️ Risk", "🚦 Signals", "🔄 Backtest"])

        # TAB 1: PORTFOLIO (5 metrics)
        with tab1:
            st.subheader("Portfolio Engine")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Portfolio Return", f"{total_port_ret * 100:.2f}%")
            c2.metric("SPY Return", f"{total_spy_ret * 100:.2f}%")
            c3.metric("Alpha vs SPY", f"{alpha_vs_spy * 100:.2f}%")
            c4.metric("Sharpe", f"{port_sharpe:.2f}")

            st.metric("Rolling Beta (SPY)", f"{latest_beta:.2f}")

            st.subheader("Allocation")
            fig_pie = px.pie(names=list(weights_dict.keys()), values=[v * 100 for v in weights_dict.values()])
            st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("Portfolio vs SPY Benchmark")
            fig_bench = go.Figure()
            fig_bench.add_trace(go.Scatter(x=port_cum.index, y=port_cum * 100, mode="lines", name="Portfolio"))
            fig_bench.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum * 100, mode="lines", name="SPY"))
            fig_bench.update_layout(height=380, yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_bench, use_container_width=True)

        # TAB 2: RISK (4 charts)
        with tab2:
            st.subheader("Risk Engine")
            st.subheader("Correlation Matrix")
            corr_df = returns_df[tickers + ["SPY"]].corr()
            fig_corr = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig_corr, use_container_width=True)

            st.subheader("Drawdown Comparison (Portfolio vs SPY)")
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=port_dd.index, y=port_dd * 100, fill="tozeroy", name="Portfolio Drawdown"))
            fig_dd.add_trace(go.Scatter(x=spy_dd.index, y=spy_dd * 100, name="SPY Drawdown"))
            fig_dd.update_layout(height=350, yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_dd, use_container_width=True)

            st.subheader("Rolling Volatility (20D Annualized) (Portfolio vs SPY)")
            fig_rvol = go.Figure()
            fig_rvol.add_trace(go.Scatter(x=roll_vol_port.index, y=roll_vol_port * 100, name="Portfolio Rolling Vol"))
            fig_rvol.add_trace(go.Scatter(x=roll_vol_spy.index, y=roll_vol_spy * 100, name="SPY Rolling Vol"))
            fig_rvol.update_layout(height=350, yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_rvol, use_container_width=True)

            st.subheader("Rolling Beta vs SPY (20D)")
            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(x=rolling_beta.index, y=rolling_beta, mode="lines", name="Rolling Beta (20D)"))
            fig_beta.add_hline(y=1.0, line_dash="dash", line_color="red")
            fig_beta.update_layout(height=350, yaxis_title="Beta")
            st.plotly_chart(fig_beta, use_container_width=True)

        # TAB 3: SIGNALS
        with tab3:
            st.subheader("Signal Engine")
            st.metric("Portfolio Signal", port_sig_label)
            fig_sig = go.Figure()
            fig_sig.add_trace(go.Scatter(x=avg_signal.index, y=avg_signal, mode="lines", name="Average Signal Score"))
            fig_sig.update_layout(height=350, yaxis_title="Signal Score (0 to 1)")
            st.plotly_chart(fig_sig, use_container_width=True)

            sig_display = sig_matrix.replace({1: "BUY", 0: "SELL"})
            st.dataframe(sig_display.tail(20), use_container_width=True)

        # TAB 4: BACKTEST
        with tab4:
            st.subheader("Backtest Engine")
            bt_df = pd.DataFrame({"Portfolio": port_cum * 100, "SPY": spy_cum * 100})
            st.dataframe(bt_df.tail(10), use_container_width=True)

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=port_cum.index, y=port_cum * 100, name="Portfolio"))
            fig_bt.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum * 100, name="SPY"))
            fig_bt.update_layout(title="Portfolio vs SPY", height=350, yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_bt, use_container_width=True)

            fig_hist = px.histogram(port_ret * 100, nbins=50, labels={"value": "Portfolio Daily Return (%)"})
            fig_hist.update_layout(title="Portfolio Returns Histogram", height=350, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
