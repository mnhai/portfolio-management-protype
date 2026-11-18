"""Bước 8 — Bull/Bear Regime Switching Strategy (myapp10.py)
Kế thừa toàn bộ tính năng từ Bước 1 -> Bước 7 (Portfolio, Optimization, Risk & Rolling Beta,
Signals, Backtest, Momentum Factors 3 biểu đồ) và bổ sung Tab 6: Regime Switching Strategy:
- Trend: MA50 so với MA200 (1 nếu MA50 > MA200, ngược lại 0).
- Momentum: Lợi nhuận 12 ngày (1 nếu > 0, ngược lại 0).
- Volatility: Biến động 20 ngày so với trung vị 60 ngày (1 nếu High Vol = Risk-off, ngược lại 0).
- Regime Score = Trend + Momentum - Volatility -> BULL / NEUTRAL / BEAR.
- Chiến lược luân chuyển phân bổ động & SPY Price Regime Map (Xanh/Vàng/Đỏ).
"""
from datetime import datetime, date
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from scipy.optimize import minimize

# 1. Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Bull/Bear Regime Switching Strategy",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Portfolio Terminal — Bull/Bear Regime Switching Strategy")
st.caption("Bước 8 — Tích hợp đầy đủ Phân tích Danh mục, Tối ưu Max Sharpe, Momentum Factors & Chiến lược Chế độ Thị trường")

# 2. Thanh bên (Sidebar)
with st.sidebar:
    st.header("⚙️ Cấu hình Danh mục")
    default_tickers = "AAPL, MSFT, NVDA"
    tickers_input = st.text_input("Mã cổ phiếu danh mục:", default_tickers)
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    benchmark_ticker = st.text_input("Benchmark chế độ (Regime Benchmark):", "SPY").strip().upper()

    start_date = st.date_input("Ngày bắt đầu:", date(2022, 1, 1), help="Nên chọn khoảng thời gian đủ dài để tính MA200")
    end_date = st.date_input("Ngày kết thúc:", datetime.today().date())
    rf_rate = st.number_input("Lãi suất phi rủi ro:", value=0.02, step=0.005, format="%.3f")
    max_weight_limit = st.slider("Giới hạn tỷ trọng tối đa (Max Weight):", 0.2, 1.0, 0.6, 0.05)

    st.subheader("⚖️ Trọng Số Danh Mục (User Weights)")
    raw_weights = {}
    default_w = 1.0 / len(tickers) if tickers else 1.0
    for t in tickers:
        raw_weights[t] = st.slider(f"Trọng số {t}:", 0.0, 1.0, float(default_w), 0.05)

    total_raw = sum(raw_weights.values())
    normalized_weights = {k: v / total_raw for k, v in raw_weights.items()} if total_raw > 0 else {k: 1.0 / len(tickers) for k in tickers}

    btn_reload = st.button("🚀 Cập nhật Terminal & Chiến Lược", use_container_width=True)


# 3. Hàm tải dữ liệu và tính toán
@st.cache_data(ttl=3600)
def fetch_terminal_data(assets: list, benchmark: str, start: str, end: str):
    symbols_to_download = list(dict.fromkeys(assets + [benchmark]))
    try:
        raw_data = yf.download(symbols_to_download, start=start, end=end, auto_adjust=True)
    except Exception as e:
        return None, f"Lỗi tải dữ liệu: {e}"

    if raw_data.empty:
        return None, "Không có dữ liệu trả về."

    close_df = raw_data["Close"].copy() if "Close" in raw_data else raw_data.copy()
    if isinstance(close_df, pd.Series):
        close_df = close_df.to_frame(name=symbols_to_download[0])
    if isinstance(close_df.columns, pd.MultiIndex):
        close_df.columns = close_df.columns.get_level_values(-1)

    close_df.dropna(how="all", inplace=True)
    close_df.ffill(inplace=True)
    close_df.dropna(inplace=True)

    if benchmark not in close_df.columns:
        return None, f"Thiếu benchmark {benchmark}."
    for a in assets:
        if a not in close_df.columns:
            return None, f"Thiếu dữ liệu {a}."

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


def simulate_regime_switching_strategy(close_df: pd.DataFrame, assets: list, benchmark: str, weights_dict: dict, regime_df: pd.DataFrame):
    daily_returns = close_df.pct_change().dropna()
    common_idx = daily_returns.index.intersection(regime_df.index)
    daily_returns = daily_returns.loc[common_idx]
    regime_df = regime_df.loc[common_idx]

    weights_vec = np.array([weights_dict[a] for a in assets])
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
        "lagged_regime": lagged_regime
    }


def calculate_all_metrics(close_df: pd.DataFrame, assets: list, benchmark: str, user_weights: dict, rf: float, max_weight: float):
    daily_returns = close_df.pct_change().dropna()
    weights_vec = np.array([user_weights[a] for a in assets])
    user_daily = daily_returns[assets].dot(weights_vec)
    spy_daily = daily_returns[benchmark]

    user_cum = (1 + user_daily).cumprod() - 1
    spy_cum = (1 + spy_daily).cumprod() - 1
    user_vol = user_daily.std() * np.sqrt(252)
    spy_vol = spy_daily.std() * np.sqrt(252)
    user_sharpe = (user_daily.mean() * 252 - rf) / user_vol if user_vol > 0 else 0.0
    spy_sharpe = (spy_daily.mean() * 252 - rf) / spy_vol if spy_vol > 0 else 0.0

    opt_res = optimize_max_sharpe(daily_returns, assets, rf, max_weight)
    opt_vec = np.array([opt_res["weights"][a] for a in assets])
    opt_daily = daily_returns[assets].dot(opt_vec)
    opt_cum = (1 + opt_daily).cumprod() - 1

    user_wealth = (1 + user_daily).cumprod()
    user_dd = (user_wealth - user_wealth.cummax()) / user_wealth.cummax()
    opt_wealth = (1 + opt_daily).cumprod()
    opt_dd = (opt_wealth - opt_wealth.cummax()) / opt_wealth.cummax()
    spy_wealth = (1 + spy_daily).cumprod()
    spy_dd = (spy_wealth - spy_wealth.cummax()) / spy_wealth.cummax()

    rolling_vol = user_daily.rolling(20).std() * np.sqrt(252)
    spy_rolling_vol = spy_daily.rolling(20).std() * np.sqrt(252)
    rolling_beta = user_daily.rolling(20).cov(spy_daily) / spy_daily.rolling(20).var()
    latest_beta = float(rolling_beta.dropna().iloc[-1]) if not rolling_beta.dropna().empty else 1.0

    spy_price = close_df[benchmark].loc[daily_returns.index]
    synthetic_mom = (spy_price / spy_price.shift(12)) - 1
    rolling_mom_corr = user_daily.rolling(20).corr(synthetic_mom)
    latest_mom_corr = float(rolling_mom_corr.dropna().iloc[-1]) if not rolling_mom_corr.dropna().empty else 0.0

    signals = {}
    for a in assets:
        ma5 = close_df[a].rolling(5).mean()
        ma20 = close_df[a].rolling(20).mean()
        sig = "BUY" if ma5.iloc[-1] > ma20.iloc[-1] else "SELL"
        signals[a] = {"Price": close_df[a].iloc[-1], "MA5": ma5.iloc[-1], "MA20": ma20.iloc[-1], "Signal": sig}

    return {
        "user_daily": user_daily, "user_cum": user_cum, "user_total": user_cum.iloc[-1], "user_vol": user_vol, "user_sharpe": user_sharpe, "user_dd": user_dd,
        "spy_daily": spy_daily, "spy_cum": spy_cum, "spy_total": spy_cum.iloc[-1], "spy_vol": spy_vol, "spy_sharpe": spy_sharpe, "spy_dd": spy_dd,
        "opt_result": opt_res, "opt_daily": opt_daily, "opt_cum": opt_cum, "opt_total": opt_cum.iloc[-1], "opt_dd": opt_dd,
        "rolling_vol": rolling_vol, "spy_rolling_vol": spy_rolling_vol, "rolling_beta": rolling_beta, "latest_beta": latest_beta,
        "rolling_mom_corr": rolling_mom_corr, "latest_mom_corr": latest_mom_corr, "signals": signals,
        "win_rate": (user_daily > 0).mean() * 100, "daily_returns": daily_returns
    }


# 4. Hiển thị Giao diện 6 Tab
if not tickers:
    st.warning("⚠️ Vui lòng nhập danh sách mã hợp lệ.")
else:
    with st.spinner("Đang tải dữ liệu và tính toán chiến lược Regime Switching..."):
        close_data, err = fetch_terminal_data(tickers, benchmark_ticker, str(start_date), str(end_date))

    if err:
        st.error(f"⚠️ {err}")
    else:
        m = calculate_all_metrics(close_data, tickers, benchmark_ticker, normalized_weights, rf_rate, max_weight_limit)
        regime_data = compute_market_regimes(close_data[benchmark_ticker])
        sim = simulate_regime_switching_strategy(close_data, tickers, benchmark_ticker, normalized_weights, regime_data)
        opt = m["opt_result"]

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 **Portfolio & Optimization**",
            "🛡️ **Risk & Beta**",
            "📡 **Signals**",
            "🔬 **Backtest**",
            "🌪️ **Momentum Factors**",
            "🧭 **Regime Switching Strategy**"
        ])

        # TAB 1: PORTFOLIO & OPTIMIZATION
        with tab1:
            st.subheader("🎯 Danh Mục Tự Chọn vs Max Sharpe Optimization")
            col_u, col_opt, col_spy = st.columns(3)
            with col_u:
                st.info("👤 **User Portfolio**")
                st.metric("Total Return", f"{m['user_total']*100:.2f}%")
                st.metric("Volatility", f"{m['user_vol']*100:.2f}%")
                st.metric("Sharpe Ratio", f"{m['user_sharpe']:.2f}")
            with col_opt:
                st.success("🌟 **Max Sharpe Optimized**")
                st.metric("Expected Return", f"{opt['expected_return']*100:.2f}%")
                st.metric("Volatility", f"{opt['volatility']*100:.2f}%")
                st.metric("Sharpe Ratio", f"{opt['sharpe_ratio']:.2f}", delta=f"{opt['sharpe_ratio'] - m['user_sharpe']:.2f}")
            with col_spy:
                st.warning(f"🏛️ **Benchmark ({benchmark_ticker})**")
                st.metric("SPY Return", f"{m['spy_total']*100:.2f}%")
                st.metric("SPY Volatility", f"{m['spy_vol']*100:.2f}%")
                st.metric("SPY Sharpe", f"{m['spy_sharpe']:.2f}")

            col_w_bar, col_cum_chart = st.columns([1, 2])
            with col_w_bar:
                st.subheader("⚖️ So Sánh Trọng Số")
                w_df = pd.DataFrame({"User Weights": [normalized_weights[t]*100 for t in tickers], "Max Sharpe": [opt["weights"][t]*100 for t in tickers]}, index=tickers)
                fig_bar = px.bar(w_df, barmode="group", labels={"value": "Tỷ trọng (%)"})
                fig_bar.update_layout(height=380, yaxis=dict(ticksuffix="%"))
                st.plotly_chart(fig_bar, use_container_width=True)
            with col_cum_chart:
                st.subheader("📈 Lũy Kế 3 Danh Mục")
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(x=m["opt_cum"].index, y=m["opt_cum"]*100, name="Max Sharpe", line=dict(color="#00CC96", width=2.5)))
                fig_cum.add_trace(go.Scatter(x=m["user_cum"].index, y=m["user_cum"]*100, name="User Portfolio", line=dict(color="#636EFA", width=2)))
                fig_cum.add_trace(go.Scatter(x=m["spy_cum"].index, y=m["spy_cum"]*100, name=benchmark_ticker, line=dict(color="#FFA15A", dash="dash")))
                fig_cum.update_layout(height=380, yaxis=dict(ticksuffix="%"))
                st.plotly_chart(fig_cum, use_container_width=True)

        # TAB 2: RISK & BETA
        with tab2:
            st.subheader("🛡️ Quản Trị Rủi Ro & Rolling Beta (20 Ngày)")
            r1, r2, r3 = st.columns(3)
            r1.metric("⚡ Latest Rolling Beta", f"{m['latest_beta']:.2f}")
            r2.metric("📉 User Volatility", f"{m['user_vol']*100:.2f}%")
            r3.metric("⚡ User Max Drawdown", f"{m['user_dd'].min()*100:.2f}%")

            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(x=m["rolling_beta"].index, y=m["rolling_beta"], name="Rolling Beta", line=dict(color="#FFA15A")))
            fig_beta.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Market Beta = 1.0")
            fig_beta.update_layout(height=340)
            st.plotly_chart(fig_beta, use_container_width=True)

            c_dd, c_corr = st.columns(2)
            with c_dd:
                st.subheader("📉 Biểu Đồ Drawdown")
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=m["user_dd"].index, y=m["user_dd"]*100, name="User DD", fill="tozeroy", line=dict(color="#EF553B")))
                fig_dd.add_trace(go.Scatter(x=m["spy_dd"].index, y=m["spy_dd"]*100, name="SPY DD", line=dict(color="#AB63FA", dash="dot")))
                fig_dd.update_layout(height=340, yaxis=dict(ticksuffix="%"))
                st.plotly_chart(fig_dd, use_container_width=True)
            with c_corr:
                st.subheader("🔥 Ma Trận Tương Quan")
                fig_corr = px.imshow(m["daily_returns"][tickers + [benchmark_ticker]].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                fig_corr.update_layout(height=340)
                st.plotly_chart(fig_corr, use_container_width=True)

        # TAB 3: SIGNALS
        with tab3:
            st.subheader("📡 Tín Hiệu Giao Dịch MA5 / MA20")
            s_cols = st.columns(len(tickers))
            for i, t in enumerate(tickers):
                info = m["signals"][t]
                with s_cols[i]:
                    st.metric(t, f"${info['Price']:.2f}", delta="🟢 BUY" if info["Signal"] == "BUY" else "🔴 SELL")
                    st.write(f"- MA5: **${info['MA5']:.2f}** | MA20: **${info['MA20']:.2f}**")
            selected_t = st.selectbox("Chọn mã xem chi tiết biểu đồ tín hiệu:", tickers)
            fig_sig = go.Figure()
            fig_sig.add_trace(go.Scatter(x=close_data.index, y=close_data[selected_t], name="Close"))
            fig_sig.add_trace(go.Scatter(x=close_data.index, y=close_data[selected_t].rolling(5).mean(), name="MA5", line=dict(dash="dot")))
            fig_sig.add_trace(go.Scatter(x=close_data.index, y=close_data[selected_t].rolling(20).mean(), name="MA20"))
            fig_sig.update_layout(height=380, hovermode="x unified")
            st.plotly_chart(fig_sig, use_container_width=True)

        # TAB 4: BACKTEST
        with tab4:
            st.subheader("🔬 Thống Kê Lịch Sử & Histogram")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("🏆 Win Rate", f"{m['win_rate']:.1f}%")
            b2.metric("📅 Số Phiên", f"{len(m['user_daily'])}")
            b3.metric("🔼 Tốt Nhất", f"{m['user_daily'].max()*100:.2f}%")
            b4.metric("🔽 Kém Nhất", f"{m['user_daily'].min()*100:.2f}%")
            fig_h = px.histogram(pd.DataFrame({"User": m["user_daily"]*100, benchmark_ticker: m["spy_daily"]*100}), barmode="overlay", nbins=60, opacity=0.6)
            fig_h.update_layout(height=360)
            st.plotly_chart(fig_h, use_container_width=True)

        # TAB 5: MOMENTUM FACTORS (3 CHARTS)
        with tab5:
            st.subheader("🌪️ Momentum Factor Exposure (3 Biểu Đồ Phân Tích)")
            st.metric("🌀 Latest Momentum Corr", f"{m['latest_mom_corr']:.2f}")
            fig_rv = go.Figure()
            fig_rv.add_trace(go.Scatter(x=m["rolling_vol"].index, y=m["rolling_vol"]*100, name="Portfolio Rolling Vol"))
            fig_rv.add_trace(go.Scatter(x=m["spy_rolling_vol"].index, y=m["spy_rolling_vol"]*100, name="SPY Rolling Vol", line=dict(dash="dot")))
            fig_rv.update_layout(title="1. Rolling Volatility (20d)", height=320, yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_rv, use_container_width=True)

            fig_rb = go.Figure()
            fig_rb.add_trace(go.Scatter(x=m["rolling_beta"].index, y=m["rolling_beta"], name="Rolling Beta vs SPY", line=dict(color="#FFA15A")))
            fig_rb.add_hline(y=1.0, line_dash="dash", line_color="red")
            fig_rb.update_layout(title="2. Rolling Beta vs SPY (20d)", height=320)
            st.plotly_chart(fig_rb, use_container_width=True)

            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=m["rolling_mom_corr"].index, y=m["rolling_mom_corr"], name="Momentum Factor Exposure", line=dict(color="#AB63FA")))
            fig_mc.add_hline(y=0.0, line_dash="dash", line_color="gray")
            fig_mc.update_layout(title="3. Momentum Factor Exposure (Rolling Corr)", height=320, yaxis=dict(range=[-1, 1]))
            st.plotly_chart(fig_mc, use_container_width=True)

        # TAB 6: REGIME SWITCHING STRATEGY & SPY MAP
        with tab6:
            st.subheader("🧭 Chiến Lược Luân Chuyển Tự Động Theo Chế Độ Thị Trường")
            latest_row = regime_data.iloc[-1]
            current_regime = latest_row["Regime"]
            icon = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}.get(current_regime, "⚪")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🌐 Chế Độ Thị Trường", f"{icon} {current_regime}")
            c2.metric("🔢 Regime Score", f"{int(latest_row['Regime_Score'])}")
            c3.metric("🚀 Strategy Return", f"{sim['strategy_cum'].iloc[-1]*100:.2f}%", delta=f"{(sim['strategy_cum'].iloc[-1] - sim['growth_cum'].iloc[-1])*100:.2f}% vs Buy&Hold")
            c4.metric("🏛️ SPY Return", f"{sim['spy_cum'].iloc[-1]*100:.2f}%")

            fig_strat = go.Figure()
            fig_strat.add_trace(go.Scatter(x=sim["strategy_cum"].index, y=sim["strategy_cum"]*100, name="🛡️ REGIME SWITCHING", line=dict(color="#00CC96", width=3)))
            fig_strat.add_trace(go.Scatter(x=sim["growth_cum"].index, y=sim["growth_cum"]*100, name="👤 STATIC GROWTH", line=dict(color="#636EFA", width=2)))
            fig_strat.add_trace(go.Scatter(x=sim["spy_cum"].index, y=sim["spy_cum"]*100, name=f"🏛️ {benchmark_ticker}", line=dict(color="#EF553B", dash="dash")))
            fig_strat.update_layout(height=400, yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_strat, use_container_width=True)

            st.subheader("🗺️ Bản Đồ Chế Độ Thị Trường (SPY Price Regime Map)")
            fig_map = go.Figure()
            color_map = {"BULL": "#00CC96", "NEUTRAL": "#FFA15A", "BEAR": "#EF553B"}
            fig_map.add_trace(go.Scatter(x=regime_data.index, y=regime_data["Close"], mode="lines", name="SPY Price", line=dict(color="rgba(150, 150, 150, 0.4)")))
            for reg in ["BULL", "NEUTRAL", "BEAR"]:
                sub = regime_data[regime_data["Regime"] == reg]
                fig_map.add_trace(go.Scatter(x=sub.index, y=sub["Close"], mode="markers", name=f"Chế độ {reg}", marker=dict(color=color_map[reg], size=5)))
            fig_map.add_trace(go.Scatter(x=regime_data.index, y=regime_data["MA50"], name="MA50", line=dict(color="#1f77b4", dash="dot")))
            fig_map.add_trace(go.Scatter(x=regime_data.index, y=regime_data["MA200"], name="MA200", line=dict(color="#9467bd")))
            fig_map.update_layout(height=420, yaxis_title="SPY Price (USD)")
            st.plotly_chart(fig_map, use_container_width=True)

            st.subheader("📋 Chi Tiết Tín Hiệu Regime Gần Nhất")
            st.dataframe(regime_data[["Close", "Regime_Score", "Regime", "Trend_Signal", "Mom_Signal", "Vol_Risk_Off"]].tail(15), use_container_width=True)
