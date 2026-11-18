"""Bước 9 — myapp13.py: Final Bloomberg-Style Portfolio Terminal (5 tab)"""
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
def fetch_terminal_data(assets: list, benchmark: str, start: str, end: str = None):
    if end is None:
        end = str(datetime.today().date())
    factors = ["SPY", "QQQ", "IWM", "^VIX"]
    symbols = list(dict.fromkeys(assets + [benchmark] + factors))
    try:
        raw = yf.download(symbols, start=start, end=end, auto_adjust=True)
    except Exception as e:
        return None, f"Lỗi tải dữ liệu: {e}"

    if raw.empty:
        return None, "Không tìm thấy dữ liệu."

    close_df = raw["Close"].copy() if "Close" in raw else raw.copy()
    if isinstance(close_df, pd.Series):
        close_df = close_df.to_frame(name=symbols[0])
    if isinstance(close_df.columns, pd.MultiIndex):
        close_df.columns = close_df.columns.get_level_values(-1)

    close_df.rename(columns={"^VIX": "VIX"}, inplace=True)
    close_df.dropna(how="all", inplace=True)
    close_df.ffill(inplace=True)
    close_df.dropna(inplace=True)
    return close_df, None


def optimize_max_sharpe(daily_returns: pd.DataFrame, assets: list, rf: float = 0.02, max_weight: float = 1.0):
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


def run_full_quantitative_pipeline(close_df: pd.DataFrame, assets: list, benchmark: str, w_dict: dict, rf: float = 0.02, max_w: float = 1.0, n_mc: int = 5000):
    returns_df = close_df.pct_change().dropna()
    w_vec = np.array([w_dict[t] for t in assets])

    port_ret = returns_df[assets].dot(w_vec)
    spy_ret = returns_df[benchmark]

    port_wealth = (1 + port_ret).cumprod()
    spy_wealth = (1 + spy_ret).cumprod()
    port_cum = port_wealth - 1
    spy_cum = spy_wealth - 1

    total_port_ret = port_cum.iloc[-1]
    total_spy_ret = spy_cum.iloc[-1]
    alpha_vs_spy = total_port_ret - total_spy_ret

    port_vol = port_ret.std() * np.sqrt(252)
    spy_vol = spy_ret.std() * np.sqrt(252)
    port_annual_ret = port_ret.mean() * 252
    spy_annual_ret = spy_ret.mean() * 252
    port_sharpe = (port_annual_ret - rf) / port_vol if port_vol > 0 else 0.0
    spy_sharpe = (spy_annual_ret - rf) / spy_vol if spy_vol > 0 else 0.0

    port_dd = (port_wealth - port_wealth.cummax()) / port_wealth.cummax()
    spy_dd = (spy_wealth - spy_wealth.cummax()) / spy_wealth.cummax()

    roll_vol_port = port_ret.rolling(20).std() * np.sqrt(252)
    roll_vol_spy = spy_ret.rolling(20).std() * np.sqrt(252)

    # Betas
    rolling_beta = port_ret.rolling(20).cov(spy_ret) / spy_ret.rolling(20).var()
    latest_beta = float(rolling_beta.dropna().iloc[-1]) if not rolling_beta.dropna().empty else 1.0
    market_beta = float(port_ret.cov(spy_ret) / spy_ret.var()) if spy_ret.var() > 0 else 1.0

    # Synthetic Momentum
    spy_close = close_df[benchmark].loc[returns_df.index]
    synth_momentum = (spy_close / spy_close.shift(12)) - 1
    rolling_mom_corr = port_ret.rolling(20).corr(synth_momentum)
    latest_mom_exp = float(rolling_mom_corr.dropna().iloc[-1]) if not rolling_mom_corr.dropna().empty else 0.0

    # Max Sharpe Optimization
    opt = optimize_max_sharpe(returns_df, assets, rf=rf, max_weight=max_w)
    opt_vec = np.array([opt["weights"][t] for t in assets])
    opt_daily = returns_df[assets].dot(opt_vec)
    opt_cum = (1 + opt_daily).cumprod() - 1

    # Monte Carlo 5,000 portfolios
    mean_vec = returns_df[assets].mean().values * 252
    cov_mat = returns_df[assets].cov().values * 252
    mc_weights = np.random.dirichlet(np.ones(len(assets)), size=n_mc)
    mc_returns = np.dot(mc_weights, mean_vec)
    mc_vols = np.sqrt(np.einsum('ij,jk,ik->i', mc_weights, cov_mat, mc_weights))
    mc_sharpes = (mc_returns - rf) / mc_vols

    # Individual Assets Vol & Return
    indiv_vols = returns_df[assets].std().values * np.sqrt(252)
    indiv_rets = returns_df[assets].mean().values * 252

    # CAPM Alpha & Treynor Ratio
    capm_expected = rf + market_beta * (spy_annual_ret - rf)
    capm_alpha = port_annual_ret - capm_expected
    treynor_ratio = (port_annual_ret - rf) / market_beta if market_beta != 0 else 0.0

    # Factor Betas (SPY, QQQ, IWM, VIX)
    factor_names = [f for f in ["SPY", "QQQ", "IWM", "VIX"] if f in returns_df.columns]
    factor_betas = {}
    for f in factor_names:
        var_f = returns_df[f].var()
        factor_betas[f] = float(port_ret.cov(returns_df[f]) / var_f) if var_f > 0 else 0.0

    # Market Regime RISK ON / RISK OFF từ SPY, QQQ, VIX
    if "SPY" in close_df.columns and "QQQ" in close_df.columns and "VIX" in close_df.columns:
        spy_above_ma50 = close_df["SPY"].iloc[-1] > close_df["SPY"].rolling(50).mean().iloc[-1]
        qqq_above_ma50 = close_df["QQQ"].iloc[-1] > close_df["QQQ"].rolling(50).mean().iloc[-1]
        vix_calm = close_df["VIX"].iloc[-1] < close_df["VIX"].rolling(50).mean().iloc[-1]
        risk_regime = "RISK ON" if (spy_above_ma50 and qqq_above_ma50 and vix_calm) else "RISK OFF"
    else:
        risk_regime = "RISK ON" if latest_beta >= 1.0 else "RISK OFF"

    # Signals
    sig_matrix = pd.DataFrame(index=close_df.index)
    for t in assets:
        ma5 = close_df[t].rolling(5).mean()
        ma20 = close_df[t].rolling(20).mean()
        sig_matrix[t] = np.where(ma5 > ma20, 1, 0)
    avg_signal = sig_matrix.mean(axis=1)
    port_sig_label = "BUY" if avg_signal.iloc[-1] >= 0.5 else "SELL"

    # Bull/Neutral/Bear Regime Switching Strategy
    ma50_spy = spy_close.rolling(50).mean()
    ma200_spy = spy_close.rolling(200).mean()
    t_sig = np.where(ma50_spy > ma200_spy, 1, 0)
    m_sig = np.where(synth_momentum > 0, 1, 0)
    v_sig = np.where(roll_vol_spy > roll_vol_spy.rolling(60).median(), 1, 0)
    r_score = pd.Series(t_sig + m_sig - v_sig, index=returns_df.index)
    r_labels = pd.Series(
        np.select([r_score >= 1, r_score == 0, r_score <= -1], ["BULL", "NEUTRAL", "BEAR"], default="NEUTRAL"),
        index=returns_df.index
    )
    lagged_r = r_labels.shift(1).fillna("NEUTRAL")
    strat_ret = pd.Series(
        np.where(lagged_r == "BULL", port_ret, np.where(lagged_r == "NEUTRAL", 0.5 * port_ret + 0.5 * spy_ret, spy_ret)),
        index=returns_df.index
    )
    strat_cum = (1 + strat_ret).cumprod() - 1
    regime_df = pd.DataFrame({"Date": returns_df.index, "SPY": spy_close.values, "Regime_Score": r_score.values, "Regime": r_labels.values}).set_index("Date")

    # 60-Day Rolling Beta Heatmap vs SPY
    beta_60d_matrix = pd.DataFrame(index=returns_df.index)
    spy_var_60 = spy_ret.rolling(60).var()
    for t in assets:
        beta_60d_matrix[t] = returns_df[t].rolling(60).cov(spy_ret) / spy_var_60

    # Cumulative Active Return vs SPY: tích lũy (1 + (Rp - R_spy))
    cum_active_return = (1 + (port_ret - spy_ret)).cumprod() - 1

    return {
        "port_ret": port_ret, "spy_ret": spy_ret, "port_cum": port_cum, "spy_cum": spy_cum,
        "total_port_ret": total_port_ret, "total_spy_ret": total_spy_ret, "alpha_vs_spy": alpha_vs_spy,
        "port_vol": port_vol, "port_sharpe": port_sharpe, "spy_dd": spy_dd, "port_dd": port_dd,
        "roll_vol_port": roll_vol_port, "roll_vol_spy": roll_vol_spy,
        "rolling_beta": rolling_beta, "latest_beta": latest_beta, "market_beta": market_beta,
        "rolling_mom_corr": rolling_mom_corr, "latest_mom_exp": latest_mom_exp,
        "opt": opt, "opt_cum": opt_cum, "mc_vols": mc_vols, "mc_returns": mc_returns, "mc_sharpes": mc_sharpes,
        "indiv_vols": indiv_vols, "indiv_rets": indiv_rets,
        "capm_alpha": capm_alpha, "treynor_ratio": treynor_ratio, "risk_regime": risk_regime,
        "factor_betas": factor_betas, "sig_matrix": sig_matrix, "avg_signal": avg_signal, "port_sig_label": port_sig_label,
        "strat_cum": strat_cum, "regime_df": regime_df, "beta_60d_matrix": beta_60d_matrix,
        "cum_active_return": cum_active_return, "returns_df": returns_df
    }


if not tickers:
    st.warning("Vui lòng nhập ít nhất 1 mã cổ phiếu.")
else:
    close_df, err = fetch_terminal_data(tickers, "SPY", str(start_date))
    if err or close_df is None or "SPY" not in close_df.columns:
        st.error(err or "Không tải được dữ liệu danh mục hoặc SPY.")
    else:
        m = run_full_quantitative_pipeline(close_df, tickers, "SPY", weights_dict, rf=0.02, max_w=1.0, n_mc=5000)
        opt = m["opt"]

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Portfolio", "⚠️ Risk", "🚦 Signals", "🔄 Backtest", "📑 Factors"])

        # ==================== TAB 1: PORTFOLIO (13 metrics + 5 charts) ====================
        with tab1:
            st.subheader("Portfolio Engine")
            # Row 1: 4 cols
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Portfolio Return", f"{m['total_port_ret']*100:.2f}%")
            c2.metric("SPY Return", f"{m['total_spy_ret']*100:.2f}%")
            c3.metric("Alpha vs SPY", f"{m['alpha_vs_spy']*100:.2f}%")
            c4.metric("Sharpe", f"{m['port_sharpe']:.2f}")

            # Row 2: 3 cols
            r2_1, r2_2, r2_3 = st.columns(3)
            r2_1.metric("Rolling Beta (SPY)", f"{m['latest_beta']:.2f}")
            r2_2.metric("Market Beta", f"{m['market_beta']:.2f}")
            r2_3.metric("Momentum Exposure", f"{m['latest_mom_exp']:.2f}")

            # Row 3: 3 cols
            r3_1, r3_2, r3_3 = st.columns(3)
            r3_1.metric("Optimized Sharpe", f"{opt['sharpe_ratio']:.2f}")
            r3_2.metric("Optimized Return", f"{opt['expected_return']*100:.2f}%")
            r3_3.metric("Optimized Volatility", f"{opt['volatility']*100:.2f}%")

            # Row 4: 3 cols
            r4_1, r4_2, r4_3 = st.columns(3)
            r4_1.metric("CAPM Alpha", f"{m['capm_alpha']*100:.2f}%")
            r4_2.metric("Treynor Ratio", f"{m['treynor_ratio']:.2f}")
            r4_3.metric("Regime", m["risk_regime"])

            # Chart 1: Pie chart Allocation
            st.subheader("Allocation")
            fig_pie = px.pie(names=list(weights_dict.keys()), values=[v * 100 for v in weights_dict.values()])
            st.plotly_chart(fig_pie, use_container_width=True)

            # Chart 2: Optimized Portfolio (Max Sharpe)
            st.subheader("Optimized Portfolio (Max Sharpe)")
            fig_opt_pie = px.pie(names=list(opt["weights"].keys()), values=[v * 100 for v in opt["weights"].values()], title="Optimized Allocation")
            st.plotly_chart(fig_opt_pie, use_container_width=True)

            # Chart 3: Portfolio vs SPY Benchmark
            st.subheader("Portfolio vs SPY Benchmark")
            fig_bench = go.Figure()
            fig_bench.add_trace(go.Scatter(x=m["port_cum"].index, y=m["port_cum"] * 100, mode="lines", name="Portfolio"))
            fig_bench.add_trace(go.Scatter(x=m["spy_cum"].index, y=m["spy_cum"] * 100, mode="lines", name="SPY"))
            fig_bench.update_layout(height=360, yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_bench, use_container_width=True)

            # Chart 4: Optimized vs Current vs SPY (+ scatter Individual Assets)
            st.subheader("Optimized vs Current vs SPY")
            fig_opt_comp = go.Figure()
            fig_opt_comp.add_trace(go.Scatter(x=m["port_cum"].index, y=m["port_cum"] * 100, mode="lines", name="Current Portfolio"))
            fig_opt_comp.add_trace(go.Scatter(x=m["opt_cum"].index, y=m["opt_cum"] * 100, mode="lines", name="Max Sharpe Portfolio"))
            fig_opt_comp.add_trace(go.Scatter(x=m["spy_cum"].index, y=m["spy_cum"] * 100, mode="lines", name="SPY"))
            fig_opt_comp.add_trace(go.Scatter(
                x=[m["port_cum"].index[-1]] * len(tickers),
                y=[((1 + m["returns_df"][t]).cumprod().iloc[-1] - 1) * 100 for t in tickers],
                mode="markers+text",
                text=tickers,
                textposition="top right",
                marker=dict(symbol="diamond", size=10, color="black"),
                name="Individual Assets"
            ))
            fig_opt_comp.update_layout(height=380, yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_opt_comp, use_container_width=True)

            # Chart 5: Monte Carlo Efficient Frontier (5,000 portfolios)
            st.subheader("Monte Carlo Efficient Frontier")
            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(
                x=m["mc_vols"] * 100, y=m["mc_returns"] * 100,
                mode="markers",
                marker=dict(size=4, color=m["mc_sharpes"], colorscale="Viridis", colorbar=dict(title="Sharpe"), showscale=True),
                name="Simulated Portfolios"
            ))
            fig_ef.add_trace(go.Scatter(
                x=m["indiv_vols"] * 100, y=m["indiv_rets"] * 100,
                mode="markers+text", text=tickers, textposition="top center",
                marker=dict(color="black", size=10, symbol="diamond"),
                name="Individual Assets"
            ))
            fig_ef.add_trace(go.Scatter(
                x=[opt["volatility"] * 100], y=[opt["expected_return"] * 100],
                mode="markers",
                marker=dict(color="red", size=16, symbol="star"),
                name="Max Sharpe"
            ))
            fig_ef.update_layout(height=420, xaxis_title="Volatility (%)", yaxis_title="Return (%)")
            st.plotly_chart(fig_ef, use_container_width=True)

        # ==================== TAB 2: RISK (5 charts) ====================
        with tab2:
            st.subheader("Risk Engine")
            st.subheader("Correlation Matrix")
            fig_corr = px.imshow(m["returns_df"][tickers + ["SPY"]].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig_corr, use_container_width=True)

            st.subheader("Drawdown Comparison")
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=m["port_dd"].index, y=m["port_dd"] * 100, fill="tozeroy", name="Portfolio Drawdown"))
            fig_dd.add_trace(go.Scatter(x=m["spy_dd"].index, y=m["spy_dd"] * 100, name="SPY Drawdown"))
            fig_dd.update_layout(height=340, yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_dd, use_container_width=True)

            st.subheader("Rolling Volatility (20D)")
            fig_rvol = go.Figure()
            fig_rvol.add_trace(go.Scatter(x=m["roll_vol_port"].index, y=m["roll_vol_port"] * 100, name="Portfolio Rolling Vol"))
            fig_rvol.add_trace(go.Scatter(x=m["roll_vol_spy"].index, y=m["roll_vol_spy"] * 100, name="SPY Rolling Vol"))
            fig_rvol.update_layout(height=340, yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_rvol, use_container_width=True)

            st.subheader("Rolling Beta vs SPY (20D)")
            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(x=m["rolling_beta"].index, y=m["rolling_beta"], name="Rolling Beta (20D)"))
            fig_beta.add_hline(y=1.0, line_dash="dash", line_color="red")
            fig_beta.update_layout(height=340, yaxis_title="Beta")
            st.plotly_chart(fig_beta, use_container_width=True)

            st.subheader("Momentum Factor Exposure (Rolling)")
            fig_mom = go.Figure()
            fig_mom.add_trace(go.Scatter(x=m["rolling_mom_corr"].index, y=m["rolling_mom_corr"], name="Momentum Exposure"))
            fig_mom.add_hline(y=0.0, line_dash="dash", line_color="gray")
            fig_mom.update_layout(height=340, yaxis_title="Rolling Correlation", yaxis=dict(range=[-1, 1]))
            st.plotly_chart(fig_mom, use_container_width=True)

        # ==================== TAB 3: SIGNALS ====================
        with tab3:
            st.subheader("Signal Engine")
            st.metric("Portfolio Signal", m["port_sig_label"])
            fig_sig = go.Figure()
            fig_sig.add_trace(go.Scatter(x=m["avg_signal"].index, y=m["avg_signal"], mode="lines", name="Average Signal Score"))
            fig_sig.update_layout(height=350, yaxis_title="Signal Score")
            st.plotly_chart(fig_sig, use_container_width=True)
            st.dataframe(m["sig_matrix"].replace({1: "BUY", 0: "SELL"}).tail(20), use_container_width=True)

        # ==================== TAB 4: BACKTEST ====================
        with tab4:
            st.subheader("Backtest Engine")
            bt_df = pd.DataFrame({"Portfolio": m["port_cum"] * 100, "SPY": m["spy_cum"] * 100})
            st.dataframe(bt_df.tail(10), use_container_width=True)

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=m["port_cum"].index, y=m["port_cum"] * 100, name="Portfolio"))
            fig_bt.add_trace(go.Scatter(x=m["spy_cum"].index, y=m["spy_cum"] * 100, name="SPY"))
            fig_bt.update_layout(title="Portfolio vs SPY", height=340, yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_bt, use_container_width=True)

            fig_hist = px.histogram(m["port_ret"] * 100, nbins=50, title="Returns Histogram")
            fig_hist.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("Regime Switching Strategy")
            fig_strat = go.Figure()
            fig_strat.add_trace(go.Scatter(x=m["strat_cum"].index, y=m["strat_cum"] * 100, name="Regime Strategy"))
            fig_strat.add_trace(go.Scatter(x=m["port_cum"].index, y=m["port_cum"] * 100, name="Static Portfolio"))
            fig_strat.add_trace(go.Scatter(x=m["spy_cum"].index, y=m["spy_cum"] * 100, name="SPY"))
            fig_strat.update_layout(height=360, yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_strat, use_container_width=True)
            st.dataframe(m["regime_df"][["Regime"]].tail(20), use_container_width=True)

            st.subheader("Market Regime")
            fig_score = go.Figure()
            fig_score.add_trace(go.Scatter(x=m["regime_df"].index, y=m["regime_df"]["Regime_Score"], name="Regime Score"))
            fig_score.update_layout(height=320)
            st.plotly_chart(fig_score, use_container_width=True)
            st.dataframe(m["regime_df"][["Regime_Score", "Regime"]].tail(20), use_container_width=True)

            # Mới: SPY Price Regime Map (Bull/Neutral/Bear)
            st.subheader("SPY Price Regime Map (Bull/Neutral/Bear)")
            fig_map = go.Figure()
            cmap = {"BULL": "green", "NEUTRAL": "orange", "BEAR": "red"}
            for reg in ["BULL", "NEUTRAL", "BEAR"]:
                sub = m["regime_df"][m["regime_df"]["Regime"] == reg]
                fig_map.add_trace(go.Scatter(
                    x=sub.index, y=sub["SPY"],
                    mode="markers", name=reg,
                    marker=dict(color=cmap[reg], size=5)
                ))
            fig_map.update_layout(height=380, xaxis_title="Date", yaxis_title="SPY Price")
            st.plotly_chart(fig_map, use_container_width=True)

        # ==================== TAB 5: FACTORS (hoàn toàn mới) ====================
        with tab5:
            st.subheader("Factor Analytics")
            # Hàng metric 1 (3 cột): SPY Beta, QQQ Beta, IWM Beta
            f1_1, f1_2, f1_3 = st.columns(3)
            f1_1.metric("SPY Beta", f"{m['factor_betas'].get('SPY', 0.0):.2f}")
            f1_2.metric("QQQ Beta", f"{m['factor_betas'].get('QQQ', 0.0):.2f}")
            f1_3.metric("IWM Beta", f"{m['factor_betas'].get('IWM', 0.0):.2f}")

            # Hàng metric 2 (3 cột): VIX Beta, CAPM Alpha, Treynor Ratio
            f2_1, f2_2, f2_3 = st.columns(3)
            f2_1.metric("VIX Beta", f"{m['factor_betas'].get('VIX', 0.0):.2f}")
            f2_2.metric("CAPM Alpha", f"{m['capm_alpha']*100:.2f}%")
            f2_3.metric("Treynor Ratio", f"{m['treynor_ratio']:.2f}")

            # Metric đơn: Market Regime
            st.metric("Market Regime", m["risk_regime"])

            # 5. Factor Exposures Bar chart: Portfolio Factor Betas
            st.subheader("Portfolio Factor Betas")
            fb_df = pd.DataFrame({"Factor": list(m["factor_betas"].keys()), "Beta": list(m["factor_betas"].values())})
            fig_fb = px.bar(fb_df, x="Factor", y="Beta", text_auto=".2f")
            fig_fb.update_traces(textposition="outside")
            fig_fb.update_layout(height=360)
            st.plotly_chart(fig_fb, use_container_width=True)

            # 6. Rolling Beta Heatmap vs SPY (60-Day Rolling Beta)
            st.subheader("60-Day Rolling Beta")
            b60_clean = m["beta_60d_matrix"].dropna().iloc[-60:]
            fig_bheat = px.imshow(b60_clean.T, aspect="auto", color_continuous_scale="Viridis", labels=dict(x="Date", y="Ticker", color="Beta"))
            fig_bheat.update_layout(height=340)
            st.plotly_chart(fig_bheat, use_container_width=True)

            # 7. Active Return Attribution vs SPY: Portfolio Outperformance vs SPY
            st.subheader("Portfolio Outperformance vs SPY")
            fig_act = go.Figure()
            fig_act.add_trace(go.Scatter(x=m["cum_active_return"].index, y=m["cum_active_return"] * 100, mode="lines", name="Cumulative Active Return"))
            fig_act.update_layout(height=340, yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_act, use_container_width=True)

            # 8. Factor Summary table
            st.subheader("Factor Summary")
            st.dataframe(fb_df, use_container_width=True)

            # 9. Top/Bottom Beta Tickers từ ma trận beta 60 ngày
            latest_60d_betas = m["beta_60d_matrix"].dropna().iloc[-1].sort_values(ascending=False)
            col_top, col_bot = st.columns(2)
            with col_top:
                st.subheader("Highest Beta Names")
                st.dataframe(latest_60d_betas.head(5).rename("60D Beta"), use_container_width=True)
            with col_bot:
                st.subheader("Lowest Beta Names")
                st.dataframe(latest_60d_betas.tail(5).sort_values(ascending=True).rename("60D Beta"), use_container_width=True)
