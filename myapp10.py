"""Bước 8 — myapp10.py: Thêm Bull/Bear Regime Switching Strategy"""
from datetime import date, datetime
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from scipy.optimize import minimize

st.set_page_config(page_title="Bloomberg-Style Portfolio Terminal", page_icon="📈", layout="wide")

st.title("📈 Bloomberg-Style Portfolio Terminal")

with st.sidebar:
    st.header("⚙️ Settings")
    tickers_input = st.text_input("Tickers", "AAPL,MSFT,NVDA")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    start_date = st.date_input("Start Date", date(2022, 1, 1))

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


def optimize_max_sharpe(daily_returns: pd.DataFrame, assets: list, rf: float = 0.0, max_weight: float = 1.0):
    asset_returns = daily_returns[assets]
    mean_ret = asset_returns.mean() * 252
    cov_mat = asset_returns.cov() * 252
    num_assets = len(assets)
    w0 = np.ones(num_assets) / num_assets
    bounds = tuple((0.0, max_weight) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    def neg_sharpe(w):
        port_ret = np.dot(w, mean_ret)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
        return -(port_ret - rf) / port_vol if port_vol > 0 else 0

    res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_weights = res.x if res.success else w0
    opt_ret = np.dot(opt_weights, mean_ret)
    opt_vol = np.sqrt(np.dot(opt_weights.T, np.dot(cov_mat, opt_weights)))
    opt_sharpe = (opt_ret - rf) / opt_vol if opt_vol > 0 else 0.0
    return {"success": res.success, "weights": dict(zip(assets, opt_weights)), "expected_return": opt_ret, "volatility": opt_vol, "sharpe_ratio": opt_sharpe}


def compute_market_regimes(spy_close: pd.Series):
    spy_daily = spy_close.pct_change()
    ma50 = spy_close.rolling(50).mean()
    ma200 = spy_close.rolling(200).mean()
    trend_signal = np.where(ma50 > ma200, 1, 0)
    mom12 = (spy_close / spy_close.shift(12)) - 1
    mom_signal = np.where(mom12 > 0, 1, 0)
    vol20 = spy_daily.rolling(20).std() * np.sqrt(252)
    vol_median60 = vol20.rolling(60).median()
    vol_risk_off = np.where(vol20 > vol_median60, 1, 0)
    regime_score = trend_signal + mom_signal - vol_risk_off

    conditions = [regime_score >= 1, regime_score == 0, regime_score <= -1]
    choices = ["BULL", "NEUTRAL", "BEAR"]
    regime_labels = pd.Series(np.select(conditions, choices, default="NEUTRAL"), index=spy_close.index)

    return pd.DataFrame({
        "Close": spy_close, "MA50": ma50, "MA200": ma200, "Trend_Signal": trend_signal,
        "Mom12": mom12, "Mom_Signal": mom_signal, "Vol20": vol20, "Vol_Median60": vol_median60,
        "Vol_Risk_Off": vol_risk_off, "Regime_Score": regime_score, "Regime": regime_labels
    }).dropna()


def simulate_regime_switching_strategy(close_df: pd.DataFrame, assets: list, benchmark: str, w_dict: dict, regime_df: pd.DataFrame):
    daily_returns = close_df.pct_change().dropna()
    common_idx = daily_returns.index.intersection(regime_df.index)
    daily_returns = daily_returns.loc[common_idx]
    regime_df = regime_df.loc[common_idx]

    weights_vec = np.array([w_dict[a] for a in assets])
    growth_daily = daily_returns[assets].dot(weights_vec)
    defensive_daily = daily_returns[benchmark]

    lagged_regime = regime_df["Regime"].shift(1).fillna("NEUTRAL")
    strategy_daily = pd.Series(index=common_idx, dtype=float)
    for t in common_idx:
        reg = lagged_regime.loc[t]
        if reg == "BULL":
            strategy_daily.loc[t] = growth_daily.loc[t]
        elif reg == "NEUTRAL":
            strategy_daily.loc[t] = 0.5 * growth_daily.loc[t] + 0.5 * defensive_daily.loc[t]
        else:
            strategy_daily.loc[t] = defensive_daily.loc[t]

    return {
        "strategy_daily": strategy_daily,
        "strategy_cum": (1 + strategy_daily).cumprod() - 1,
        "growth_daily": growth_daily,
        "growth_cum": (1 + growth_daily).cumprod() - 1,
        "spy_daily": defensive_daily,
        "spy_cum": (1 + defensive_daily).cumprod() - 1,
        "regime_df": regime_df,
    }


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

        # Rolling Beta & Market Beta
        rolling_beta = port_ret.rolling(20).cov(spy_ret) / spy_ret.rolling(20).var()
        latest_beta = float(rolling_beta.dropna().iloc[-1]) if not rolling_beta.dropna().empty else 1.0
        market_beta = float(port_ret.cov(spy_ret) / spy_ret.var()) if spy_ret.var() > 0 else 1.0

        # Synthetic Momentum Factor
        spy_close = close_df["SPY"].loc[returns_df.index]
        synth_momentum = (spy_close / spy_close.shift(12)) - 1
        rolling_mom_corr = port_ret.rolling(20).corr(synth_momentum)
        latest_mom_exp = float(rolling_mom_corr.dropna().iloc[-1]) if not rolling_mom_corr.dropna().empty else 0.0

        # Max Sharpe Optimization
        opt = optimize_max_sharpe(returns_df, tickers, rf=0.0, max_weight=1.0)
        opt_vec = np.array([opt["weights"][t] for t in tickers])
        opt_daily = returns_df[tickers].dot(opt_vec)
        opt_cum = (1 + opt_daily).cumprod() - 1

        # Signals
        sig_matrix = pd.DataFrame(index=close_df.index)
        for t in tickers:
            ma5 = close_df[t].rolling(5).mean()
            ma20 = close_df[t].rolling(20).mean()
            sig_matrix[t] = np.where(ma5 > ma20, 1, 0)
        avg_signal = sig_matrix.mean(axis=1)
        port_sig_label = "BUY" if avg_signal.iloc[-1] >= 0.5 else "SELL"

        # Market Regimes & Strategy
        regime_data = compute_market_regimes(close_df["SPY"])
        sim = simulate_regime_switching_strategy(close_df, tickers, "SPY", weights_dict, regime_data)

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio", "⚠️ Risk", "🚦 Signals", "🔄 Backtest"])

        # TAB 1: PORTFOLIO (10 metrics)
        with tab1:
            st.subheader("Portfolio Engine")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Portfolio Return", f"{total_port_ret * 100:.2f}%")
            c2.metric("SPY Return", f"{total_spy_ret * 100:.2f}%")
            c3.metric("Alpha vs SPY", f"{alpha_vs_spy * 100:.2f}%")
            c4.metric("Sharpe", f"{port_sharpe:.2f}")

            r2_1, r2_2, r2_3 = st.columns(3)
            r2_1.metric("Rolling Beta (SPY)", f"{latest_beta:.2f}")
            r2_2.metric("Market Beta", f"{market_beta:.2f}")
            r2_3.metric("Momentum Exposure", f"{latest_mom_exp:.2f}")

            r3_1, r3_2, r3_3 = st.columns(3)
            r3_1.metric("Optimized Sharpe", f"{opt['sharpe_ratio']:.2f}")
            r3_2.metric("Optimized Return", f"{opt['expected_return'] * 100:.2f}%")
            r3_3.metric("Optimized Volatility", f"{opt['volatility'] * 100:.2f}%")

            st.subheader("Allocation")
            fig_pie = px.pie(names=list(weights_dict.keys()), values=[v * 100 for v in weights_dict.values()])
            st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("Optimized Portfolio (Max Sharpe)")
            fig_opt_pie = px.pie(names=list(opt["weights"].keys()), values=[v * 100 for v in opt["weights"].values()], title="Optimized Allocation")
            st.plotly_chart(fig_opt_pie, use_container_width=True)

            st.subheader("Portfolio vs SPY Benchmark")
            fig_bench = go.Figure()
            fig_bench.add_trace(go.Scatter(x=port_cum.index, y=port_cum * 100, mode="lines", name="Portfolio"))
            fig_bench.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum * 100, mode="lines", name="SPY"))
            fig_bench.update_layout(height=380, yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_bench, use_container_width=True)

            st.subheader("Optimized vs Current vs SPY")
            fig_opt_comp = go.Figure()
            fig_opt_comp.add_trace(go.Scatter(x=port_cum.index, y=port_cum * 100, mode="lines", name="Current Portfolio"))
            fig_opt_comp.add_trace(go.Scatter(x=opt_cum.index, y=opt_cum * 100, mode="lines", name="Max Sharpe Portfolio"))
            fig_opt_comp.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum * 100, mode="lines", name="SPY"))
            fig_opt_comp.update_layout(height=380, yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_opt_comp, use_container_width=True)

        # TAB 2: RISK (5 charts)
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

            st.subheader("Momentum Factor Exposure (Rolling)")
            fig_mom = go.Figure()
            fig_mom.add_trace(go.Scatter(x=rolling_mom_corr.index, y=rolling_mom_corr, mode="lines", name="Momentum Exposure"))
            fig_mom.add_hline(y=0.0, line_dash="dash", line_color="gray")
            fig_mom.update_layout(height=350, yaxis_title="Rolling Correlation", yaxis=dict(range=[-1, 1]))
            st.plotly_chart(fig_mom, use_container_width=True)

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

        # TAB 4: BACKTEST (cũ + Regime Switching Strategy & Market Regime)
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

            st.subheader("Regime Switching Strategy")
            fig_strat = go.Figure()
            fig_strat.add_trace(go.Scatter(x=sim["strategy_cum"].index, y=sim["strategy_cum"] * 100, name="Regime Strategy"))
            fig_strat.add_trace(go.Scatter(x=sim["growth_cum"].index, y=sim["growth_cum"] * 100, name="Static Portfolio"))
            fig_strat.add_trace(go.Scatter(x=sim["spy_cum"].index, y=sim["spy_cum"] * 100, name="SPY"))
            fig_strat.update_layout(height=380, yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_strat, use_container_width=True)

            reg_table = sim["regime_df"][["Regime"]].tail(20)
            st.dataframe(reg_table, use_container_width=True)

            st.subheader("Market Regime")
            fig_score = go.Figure()
            fig_score.add_trace(go.Scatter(x=regime_data.index, y=regime_data["Regime_Score"], mode="lines", name="Regime Score"))
            fig_score.update_layout(height=320, yaxis_title="Regime Score")
            st.plotly_chart(fig_score, use_container_width=True)

            st.dataframe(sim["regime_df"][["Regime_Score", "Regime"]].tail(20), use_container_width=True)
