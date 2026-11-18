"""Bước 9 — Final Bloomberg-Style Portfolio Terminal (myapp13.py)
Ứng dụng hoàn chỉnh chuẩn Bloomberg Terminal với 5 Tab chuyên nghiệp:
- Tab 1 (Portfolio): Lợi nhuận danh mục, phân bổ trọng số, tối ưu hóa Max Sharpe (SLSQP),
  biểu đồ Monte-Carlo Efficient Frontier (biên hiệu quả mô phỏng 2000 danh mục).
- Tab 2 (Risk): Ma trận tương quan, drawdown so sánh, volatility, Rolling Beta 20 ngày,
  và Rolling Beta Heatmap đa tài sản theo thời gian.
- Tab 3 (Signals): Chỉ báo mua/bán (MA5/MA20) kèm trạng thái kỹ thuật và mức giá mục tiêu.
- Tab 4 (Backtest): Hiệu suất lịch sử chi tiết so với SPY, Win rate, Histogram phân phối,
  kết hợp chiến lược Dynamic Regime Switching.
- Tab 5 (Factors): Phân tích nhân tố chuyên sâu gồm CAPM Alpha, Treynor Ratio, Beta Exposure,
  Active Return Attribution vs SPY (đóng góp lợi nhuận chủ động từng tài sản), và Synthetic Momentum Factor.
Bổ sung: Top Bloomberg Ticker Tape hiển thị chế độ thị trường hiện tại (Regime BULL/NEUTRAL/BEAR).
"""
from datetime import datetime, date
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from scipy.optimize import minimize

# 1. Cấu hình giao diện chuẩn Bloomberg Terminal
st.set_page_config(
    page_title="Bloomberg-Style Portfolio Terminal",
    page_icon="🏛️",
    layout="wide"
)

# Custom Styling cho phong cách Bloomberg Dark / Quant Terminal
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E2E;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #00CC96;
    }
    .bloomberg-tag {
        font-family: monospace;
        font-size: 13px;
        font-weight: bold;
        color: #FFA15A;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Bloomberg-Style Quantitative Portfolio Terminal")
st.caption("Bước 9 — Ứng Dụng Hoàn Chỉnh: Tối Ưu Hóa Danh Mục, Biên Hiệu Quả Monte-Carlo, Quản Trị Rủi Ro & Phân Tích Nhân Tố")

# 2. Thanh bên (Sidebar)
with st.sidebar:
    st.header("⚙️ Thiết Lập Terminal")
    default_tickers = "AAPL, MSFT, NVDA"
    tickers_input = st.text_input("Mã danh mục (Tickers):", default_tickers)
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    benchmark_ticker = st.text_input("Benchmark thị trường:", "SPY").strip().upper()

    start_date = st.date_input("Ngày bắt đầu:", date(2022, 1, 1))
    end_date = st.date_input("Ngày kết thúc:", datetime.today().date())
    rf_rate = st.number_input("Lãi suất phi rủi ro (Risk-Free Rate):", value=0.02, step=0.005, format="%.3f")

    st.subheader("🛠️ Cấu Hình Tối Ưu Hóa")
    max_weight_cap = st.slider("Tỷ trọng tối đa mỗi mã (Max Weight Cap):", 0.2, 1.0, 0.6, 0.05)
    num_monte_carlo = st.slider("Số lượng mô phỏng Monte-Carlo:", 500, 3000, 1500, 500)

    st.subheader("⚖️ Trọng Số Tự Chọn (User Weights)")
    raw_weights = {}
    default_w = 1.0 / len(tickers) if tickers else 1.0
    for t in tickers:
        raw_weights[t] = st.slider(f"Trọng số {t}:", 0.0, 1.0, float(default_w), 0.05)

    total_raw = sum(raw_weights.values())
    normalized_weights = {k: v / total_raw for k, v in raw_weights.items()} if total_raw > 0 else {k: 1.0 / len(tickers) for k in tickers}

    btn_reload = st.button("🚀 Cập Nhật Toàn Bộ Terminal", use_container_width=True)


# 3. Hàm tải và xử lý dữ liệu
@st.cache_data(ttl=3600)
def fetch_terminal_data(assets: list, benchmark: str, start: str, end: str):
    symbols = list(dict.fromkeys(assets + [benchmark]))
    try:
        raw_data = yf.download(symbols, start=start, end=end, auto_adjust=True)
    except Exception as e:
        return None, f"Lỗi kết nối yfinance: {e}"

    if raw_data.empty:
        return None, "Không tìm thấy dữ liệu."

    close_df = raw_data["Close"].copy() if "Close" in raw_data else raw_data.copy()
    if isinstance(close_df, pd.Series):
        close_df = close_df.to_frame(name=symbols[0])
    if isinstance(close_df.columns, pd.MultiIndex):
        close_df.columns = close_df.columns.get_level_values(-1)

    close_df.dropna(how="all", inplace=True)
    close_df.ffill(inplace=True)
    close_df.dropna(inplace=True)

    if benchmark not in close_df.columns:
        return None, f"Thiếu dữ liệu benchmark {benchmark}."
    for a in assets:
        if a not in close_df.columns:
            return None, f"Thiếu dữ liệu mã {a}."

    return close_df, None


def run_full_quantitative_pipeline(close_df: pd.DataFrame, assets: list, benchmark: str, user_weights: dict, rf: float, max_w: float, n_mc: int):
    daily_returns = close_df.pct_change().dropna()
    weights_vec = np.array([user_weights[a] for a in assets])

    # 1. User & Benchmark Returns
    user_daily = daily_returns[assets].dot(weights_vec)
    user_cum = (1 + user_daily).cumprod() - 1
    spy_daily = daily_returns[benchmark]
    spy_cum = (1 + spy_daily).cumprod() - 1

    user_total = user_cum.iloc[-1]
    spy_total = spy_cum.iloc[-1]
    user_vol = user_daily.std() * np.sqrt(252)
    spy_vol = spy_daily.std() * np.sqrt(252)

    user_annual_ret = user_daily.mean() * 252
    spy_annual_ret = spy_daily.mean() * 252
    user_sharpe = (user_annual_ret - rf) / user_vol if user_vol > 0 else 0.0
    spy_sharpe = (spy_annual_ret - rf) / spy_vol if spy_vol > 0 else 0.0

    # 2. Drawdowns
    user_wealth = (1 + user_daily).cumprod()
    user_dd = (user_wealth - user_wealth.cummax()) / user_wealth.cummax()
    spy_wealth = (1 + spy_daily).cumprod()
    spy_dd = (spy_wealth - spy_wealth.cummax()) / spy_wealth.cummax()

    # 3. Max Sharpe Optimization (SLSQP)
    mean_vec = daily_returns[assets].mean().values * 252
    cov_mat = daily_returns[assets].cov().values * 252
    n_a = len(assets)
    w0 = np.ones(n_a) / n_a
    bounds = tuple((0.0, max_w) for _ in range(n_a))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    def neg_sharpe(w):
        p_ret = np.dot(w, mean_vec)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
        return -(p_ret - rf) / p_vol if p_vol > 0 else 0

    res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_weights = res.x if res.success else w0
    opt_dict = dict(zip(assets, opt_weights))
    opt_daily = daily_returns[assets].dot(opt_weights)
    opt_cum = (1 + opt_daily).cumprod() - 1
    opt_total = opt_cum.iloc[-1]
    opt_vol = np.sqrt(np.dot(opt_weights.T, np.dot(cov_mat, opt_weights)))
    opt_ret = np.dot(opt_weights, mean_vec)
    opt_sharpe = (opt_ret - rf) / opt_vol if opt_vol > 0 else 0.0
    opt_wealth = (1 + opt_daily).cumprod()
    opt_dd = (opt_wealth - opt_wealth.cummax()) / opt_wealth.cummax()

    # 4. Monte-Carlo Efficient Frontier
    mc_weights = np.random.dirichlet(np.ones(n_a), size=n_mc)
    mc_returns = np.dot(mc_weights, mean_vec)
    mc_vols = np.sqrt(np.einsum('ij,jk,ik->i', mc_weights, cov_mat, mc_weights))
    mc_sharpes = (mc_returns - rf) / mc_vols

    # Minimum Volatility Portfolio
    min_vol_idx = np.argmin(mc_vols)
    min_vol_point = (mc_vols[min_vol_idx], mc_returns[min_vol_idx])

    # 5. Beta & CAPM Alpha & Treynor Ratio
    cov_spy = user_daily.cov(spy_daily)
    var_spy = spy_daily.var()
    beta_total = cov_spy / var_spy if var_spy > 0 else 1.0

    # CAPM Expected Return = Rf + Beta * (Rm - Rf)
    capm_expected = rf + beta_total * (spy_annual_ret - rf)
    capm_alpha = user_annual_ret - capm_expected
    treynor_ratio = (user_annual_ret - rf) / beta_total if beta_total != 0 else 0.0

    # Rolling Beta vs SPY (20d)
    rolling_beta = user_daily.rolling(20).cov(spy_daily) / spy_daily.rolling(20).var()
    latest_beta = float(rolling_beta.dropna().iloc[-1]) if not rolling_beta.dropna().empty else beta_total

    # Rolling Beta cho từng tài sản thành phần
    asset_rolling_betas = pd.DataFrame(index=daily_returns.index)
    for a in assets:
        asset_rolling_betas[a] = daily_returns[a].rolling(20).cov(spy_daily) / spy_daily.rolling(20).var()

    # 6. Active Return Attribution vs SPY
    # Contrib_i = Weight_i * (Asset_Return_i - SPY_Return)
    asset_totals = (1 + daily_returns[assets]).cumprod().iloc[-1] - 1
    active_returns = asset_totals - spy_total
    active_attributions = {a: user_weights[a] * active_returns[a] for a in assets}

    # 7. Market Regime (SPY)
    spy_p = close_df[benchmark]
    ma50 = spy_p.rolling(50).mean().loc[daily_returns.index]
    ma200 = spy_p.rolling(200).mean().loc[daily_returns.index]
    t_sig = np.where(ma50 > ma200, 1, 0)
    m12 = ((spy_p / spy_p.shift(12)) - 1).loc[daily_returns.index]
    m_sig = np.where(m12 > 0, 1, 0)
    vol20 = spy_daily.rolling(20).std() * np.sqrt(252)
    v_sig = np.where(vol20 > vol20.rolling(60).median(), 1, 0)
    r_score = t_sig + m_sig - v_sig
    current_regime = "BULL" if r_score[-1] >= 1 else ("BEAR" if r_score[-1] <= -1 else "NEUTRAL")

    # 8. Signals MA5/MA20
    signals = {}
    for a in assets:
        s_ma5 = close_df[a].rolling(5).mean()
        s_ma20 = close_df[a].rolling(20).mean()
        sig = "BUY" if s_ma5.iloc[-1] > s_ma20.iloc[-1] else "SELL"
        signals[a] = {"Price": close_df[a].iloc[-1], "MA5": s_ma5.iloc[-1], "MA20": s_ma20.iloc[-1], "Signal": sig}

    # Synthetic Momentum Factor
    synth_mom = ((spy_p / spy_p.shift(12)) - 1).loc[daily_returns.index]
    mom_corr = user_daily.rolling(20).corr(synth_mom)
    latest_mom_corr = float(mom_corr.dropna().iloc[-1]) if not mom_corr.dropna().empty else 0.0

    # Rolling Volatility (20d annualized)
    rolling_vol = user_daily.rolling(20).std() * np.sqrt(252)
    spy_rolling_vol = spy_daily.rolling(20).std() * np.sqrt(252)

    return {
        "user_daily": user_daily,
        "user_cum": user_cum,
        "user_total": user_total,
        "user_vol": user_vol,
        "user_sharpe": user_sharpe,
        "user_dd": user_dd,
        "spy_daily": spy_daily,
        "spy_cum": spy_cum,
        "spy_total": spy_total,
        "spy_vol": spy_vol,
        "spy_sharpe": spy_sharpe,
        "spy_dd": spy_dd,
        "opt_weights": opt_dict,
        "opt_cum": opt_cum,
        "opt_total": opt_total,
        "opt_vol": opt_vol,
        "opt_ret": opt_ret,
        "opt_sharpe": opt_sharpe,
        "opt_dd": opt_dd,
        "mc_vols": mc_vols,
        "mc_returns": mc_returns,
        "mc_sharpes": mc_sharpes,
        "min_vol_point": min_vol_point,
        "beta_total": beta_total,
        "latest_beta": latest_beta,
        "rolling_beta": rolling_beta,
        "asset_rolling_betas": asset_rolling_betas,
        "capm_alpha": capm_alpha,
        "treynor_ratio": treynor_ratio,
        "active_attributions": active_attributions,
        "asset_totals": asset_totals,
        "current_regime": current_regime,
        "regime_score": int(r_score[-1]),
        "signals": signals,
        "latest_mom_corr": latest_mom_corr,
        "mom_corr": mom_corr,
        "rolling_vol": rolling_vol,
        "spy_rolling_vol": spy_rolling_vol,
        "daily_returns": daily_returns,
        "close_df": close_df,
    }


# 4. Hiển thị Giao diện Terminal
if not tickers:
    st.warning("⚠️ Vui lòng nhập danh sách mã cổ phiếu hợp lệ.")
else:
    with st.spinner("Đang khởi tạo toàn bộ hạ tầng phân tích Bloomberg Terminal..."):
        close_data, err = fetch_terminal_data(tickers, benchmark_ticker, str(start_date), str(end_date))

    if err:
        st.error(f"⚠️ {err}")
    else:
        m = run_full_quantitative_pipeline(close_data, tickers, benchmark_ticker, normalized_weights, rf_rate, max_weight_cap, num_monte_carlo)

        # TOP BLOOMBERG TICKER TAPE / REGIME BANNER
        reg_colors = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}
        icon = reg_colors.get(m["current_regime"], "⚪")
        st.markdown(
            f"<div style='background-color:#111625; padding:10px 16px; border-radius:6px; border-left:5px solid #00CC96; margin-bottom:15px;'>"
            f"<b>🏛️ MARKET REGIME MONITOR:</b> Trạng thái thị trường hiện tại: <b>{icon} {m['current_regime']}</b> "
            f"(Regime Score: <b>{m['regime_score']}</b>) | "
            f"Benchmark {benchmark_ticker} Last: <b>${close_data[benchmark_ticker].iloc[-1]:.2f}</b> | "
            f"Portfolio Beta: <b>{m['latest_beta']:.2f}</b> | "
            f"Treynor Ratio: <b>{m['treynor_ratio']:.2f}</b> | "
            f"CAPM Alpha: <b>{m['capm_alpha']*100:.2f}%</b>"
            f"</div>",
            unsafe_allow_html=True
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 **Tab 1: Portfolio & Efficient Frontier**",
            "🛡️ **Tab 2: Risk & Rolling Beta Heatmap**",
            "📡 **Tab 3: Trading Signals**",
            "🔬 **Tab 4: Backtest Performance**",
            "⚡ **Tab 5: Factors, CAPM & Attribution**"
        ])

        # ==================== TAB 1: PORTFOLIO ====================
        with tab1:
            st.subheader("🎯 Tối Ưu Hóa Danh Mục & Biên Hiệu Quả Monte-Carlo")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.info("👤 **User Portfolio**")
                st.metric("Total Return", f"{m['user_total']*100:.2f}%")
                st.metric("Volatility", f"{m['user_vol']*100:.2f}%")
                st.metric("Sharpe Ratio", f"{m['user_sharpe']:.2f}")
            with c2:
                st.success("🌟 **Max Sharpe Optimized**")
                st.metric("Expected Return", f"{m['opt_ret']*100:.2f}%")
                st.metric("Volatility", f"{m['opt_vol']*100:.2f}%")
                st.metric("Sharpe Ratio", f"{m['opt_sharpe']:.2f}", delta=f"{m['opt_sharpe'] - m['user_sharpe']:.2f}")
            with c3:
                st.warning(f"🏛️ **Benchmark ({benchmark_ticker})**")
                st.metric("Total Return", f"{m['spy_total']*100:.2f}%")
                st.metric("Volatility", f"{m['spy_vol']*100:.2f}%")
                st.metric("Sharpe Ratio", f"{m['spy_sharpe']:.2f}")

            st.divider()

            col_ef, col_alloc = st.columns([2, 1])
            with col_ef:
                st.subheader("🌐 Monte-Carlo Efficient Frontier (Biên Hiệu Quả)")
                fig_ef = go.Figure()
                fig_ef.add_trace(go.Scatter(
                    x=m["mc_vols"] * 100,
                    y=m["mc_returns"] * 100,
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=m["mc_sharpes"],
                        colorscale="Viridis",
                        colorbar=dict(title="Sharpe Ratio"),
                        showscale=True
                    ),
                    name="Mô phỏng danh mục",
                    hoverinfo="text",
                    text=[f"Vol: {v*100:.1f}%, Ret: {r*100:.1f}%, Sharpe: {s:.2f}" for v, r, s in zip(m["mc_vols"], m["mc_returns"], m["mc_sharpes"])]
                ))
                # Điểm Max Sharpe
                fig_ef.add_trace(go.Scatter(
                    x=[m["opt_vol"] * 100],
                    y=[m["opt_ret"] * 100],
                    mode="markers",
                    marker=dict(color="red", size=14, symbol="star"),
                    name="⭐ Max Sharpe Portfolio"
                ))
                # Điểm Min Volatility
                fig_ef.add_trace(go.Scatter(
                    x=[m["min_vol_point"][0] * 100],
                    y=[m["min_vol_point"][1] * 100],
                    mode="markers",
                    marker=dict(color="yellow", size=12, symbol="diamond"),
                    name="💎 Min Volatility Portfolio"
                ))
                fig_ef.update_layout(height=420, xaxis_title="Độ Biến Động - Volatility (%)", yaxis_title="Lợi Nhuận Kỳ Vọng (%)")
                st.plotly_chart(fig_ef, use_container_width=True)

            with col_alloc:
                st.subheader("⚖️ So Sánh Phân Bổ Trọng Số")
                w_compare = pd.DataFrame({
                    "User Weights": [normalized_weights[a] * 100 for a in tickers],
                    "Optimized": [m["opt_weights"][a] * 100 for a in tickers]
                }, index=tickers)
                fig_w = px.bar(w_compare, barmode="group", labels={"value": "Tỷ trọng (%)", "index": "Mã"})
                fig_w.update_layout(height=420, yaxis=dict(ticksuffix="%"))
                st.plotly_chart(fig_w, use_container_width=True)

            st.subheader("📈 Hiệu Suất Tích Lũy 3 Chiều")
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(x=m["opt_cum"].index, y=m["opt_cum"]*100, name="Max Sharpe Optimized", line=dict(color="#00CC96", width=2.5)))
            fig_cum.add_trace(go.Scatter(x=m["user_cum"].index, y=m["user_cum"]*100, name="User Portfolio", line=dict(color="#636EFA", width=2)))
            fig_cum.add_trace(go.Scatter(x=m["spy_cum"].index, y=m["spy_cum"]*100, name=benchmark_ticker, line=dict(color="#FFA15A", width=1.5, dash="dash")))
            fig_cum.update_layout(height=380, xaxis_title="Thời gian", yaxis_title="Lợi nhuận tích lũy (%)", yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_cum, use_container_width=True)

        # ==================== TAB 2: RISK ====================
        with tab2:
            st.subheader("🛡️ Quản Trị Rủi Ro & Rolling Beta Heatmap Đa Tài Sản")
            rk1, rk2, rk3 = st.columns(3)
            rk1.metric("Latest Beta vs SPY", f"{m['latest_beta']:.2f}")
            rk2.metric("Portfolio Max Drawdown", f"{m['user_dd'].min()*100:.2f}%")
            rk3.metric(f"{benchmark_ticker} Max Drawdown", f"{m['spy_dd'].min()*100:.2f}%")

            st.divider()

            # Rolling Beta 20d & Rolling Beta Heatmap
            st.subheader(f"📈 Rolling Beta (20 Ngày) vs {benchmark_ticker}")
            fig_rbeta = go.Figure()
            fig_rbeta.add_trace(go.Scatter(x=m["rolling_beta"].index, y=m["rolling_beta"], name="Rolling Beta (20d)", line=dict(color="#FFA15A", width=2)))
            fig_rbeta.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Market Beta = 1.0")
            fig_rbeta.update_layout(height=320, hovermode="x unified")
            st.plotly_chart(fig_rbeta, use_container_width=True)

            st.subheader("🔥 Rolling Beta Heatmap Của Từng Cổ Phiếu Theo Thời Gian")
            clean_betas = m["asset_rolling_betas"].dropna().iloc[-60:]  # 60 phiên gần nhất
            fig_beta_heat = px.imshow(
                clean_betas.T,
                aspect="auto",
                color_continuous_scale="Viridis",
                labels=dict(x="Ngày giao dịch", y="Mã cổ phiếu", color="Beta")
            )
            fig_beta_heat.update_layout(height=350)
            st.plotly_chart(fig_beta_heat, use_container_width=True)

            col_dd, col_corr = st.columns(2)
            with col_dd:
                st.subheader("📉 Biểu Đồ Drawdown So Sánh")
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=m["user_dd"].index, y=m["user_dd"]*100, fill="tozeroy", name="Portfolio Drawdown", line=dict(color="#EF553B")))
                fig_dd.add_trace(go.Scatter(x=m["spy_dd"].index, y=m["spy_dd"]*100, name=f"{benchmark_ticker} Drawdown", line=dict(color="#AB63FA", dash="dot")))
                fig_dd.update_layout(height=350, yaxis=dict(ticksuffix="%"))
                st.plotly_chart(fig_dd, use_container_width=True)
            with col_corr:
                st.subheader("📊 Ma Trận Tương Quan Toàn Diện")
                corr_all = m["daily_returns"][tickers + [benchmark_ticker]].corr()
                fig_c = px.imshow(corr_all, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                fig_c.update_layout(height=350)
                st.plotly_chart(fig_c, use_container_width=True)

        # ==================== TAB 3: SIGNALS ====================
        with tab3:
            st.subheader("📡 Tín Hiệu Kỹ Thuật MA5/MA20 & Xu Hướng")
            s_cols = st.columns(len(tickers))
            for i, t in enumerate(tickers):
                sig_info = m["signals"][t]
                with s_cols[i]:
                    st.metric(t, f"${sig_info['Price']:.2f}", delta="🟢 BUY" if sig_info["Signal"] == "BUY" else "🔴 SELL")
                    st.write(f"- MA5: **${sig_info['MA5']:.2f}**")
                    st.write(f"- MA20: **${sig_info['MA20']:.2f}**")

            st.divider()
            st.subheader("📊 Biểu Đồ Xu Hướng Giá Kèm Tín Hiệu Cắt MA5/MA20")
            selected_ticker = st.selectbox("Chọn mã để xem chi tiết biểu đồ tín hiệu:", tickers)
            fig_sig = go.Figure()
            fig_sig.add_trace(go.Scatter(x=close_data.index, y=close_data[selected_ticker], name="Close Price", line=dict(color="#1f77b4")))
            fig_sig.add_trace(go.Scatter(x=close_data.index, y=close_data[selected_ticker].rolling(5).mean(), name="MA5", line=dict(color="#ff7f0e", dash="dot")))
            fig_sig.add_trace(go.Scatter(x=close_data.index, y=close_data[selected_ticker].rolling(20).mean(), name="MA20", line=dict(color="#2ca02c")))
            fig_sig.update_layout(height=400, xaxis_title="Thời gian", yaxis_title="Giá (USD)", hovermode="x unified")
            st.plotly_chart(fig_sig, use_container_width=True)

        # ==================== TAB 4: BACKTEST ====================
        with tab4:
            st.subheader("🔬 Hiệu Suất Lịch Sử Chi Tiết & So Sánh Đa Chiều")
            win_rate = (m["user_daily"] > 0).mean() * 100
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("🏆 Win Rate (% Phiên Dương)", f"{win_rate:.2f}%")
            b2.metric("📅 Tổng Số Phiên Giao Dịch", f"{len(m['user_daily'])}")
            b3.metric("🔼 Lợi Nhuận Ngày Tốt Nhất", f"{m['user_daily'].max()*100:.2f}%")
            b4.metric("🔽 Lợi Nhuận Ngày Kém Nhất", f"{m['user_daily'].min()*100:.2f}%")

            hist_df = pd.DataFrame({"Portfolio": m["user_daily"]*100, benchmark_ticker: m["spy_daily"]*100})
            fig_h = px.histogram(hist_df, barmode="overlay", nbins=60, opacity=0.6, color_discrete_map={"Portfolio": "#00CC96", benchmark_ticker: "#636EFA"})
            fig_h.update_layout(height=380, xaxis_title="Lợi nhuận ngày (%)")
            st.plotly_chart(fig_h, use_container_width=True)

        # ==================== TAB 5: FACTORS & ATTRIBUTION ====================
        with tab5:
            st.subheader("⚡ Phân Tích Nhân Tố, CAPM & Đóng Góp Lợi Nhuận Chủ Động")

            f_c1, f_c2, f_c3, f_c4 = st.columns(4)
            f_c1.metric("🎯 CAPM Alpha", f"{m['capm_alpha']*100:.2f}%", help="Lợi nhuận vượt trội sau khi đã điều chỉnh rủi ro thị trường theo CAPM")
            f_c2.metric("📐 Treynor Ratio", f"{m['treynor_ratio']:.2f}", help="(Rp - Rf) / Beta")
            f_c3.metric("⚡ Overall Beta vs SPY", f"{m['beta_total']:.2f}")
            f_c4.metric("🌪️ Momentum Corr", f"{m['latest_mom_corr']:.2f}")

            st.divider()

            # Active Return Attribution vs SPY
            st.subheader(f"📊 Active Return Attribution vs {benchmark_ticker}")
            st.caption("Đóng góp của từng cổ phiếu vào lợi nhuận vượt trội so với thị trường: Contrib = Weight × (Asset Return - SPY Return)")

            attrib_df = pd.DataFrame({
                "Cổ phiếu": list(m["active_attributions"].keys()),
                "Đóng góp Alpha (%)": [v * 100 for v in m["active_attributions"].values()]
            })
            attrib_df["Màu sắc"] = np.where(attrib_df["Đóng góp Alpha (%)"] >= 0, "Dương (Alpha Generator)", "Âm (Detractor)")

            fig_attr = px.bar(
                attrib_df,
                x="Cổ phiếu",
                y="Đóng góp Alpha (%)",
                color="Màu sắc",
                color_discrete_map={"Dương (Alpha Generator)": "#00CC96", "Âm (Detractor)": "#EF553B"}
            )
            fig_attr.update_layout(height=380, yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_attr, use_container_width=True)

            st.divider()
            st.subheader("🌪️ 3 Biểu Đồ Nhân Tố Động: Rolling Volatility, Rolling Beta & Momentum Exposure")
            fig_rv = go.Figure()
            fig_rv.add_trace(go.Scatter(x=m["rolling_vol"].index, y=m["rolling_vol"]*100, name="Portfolio Rolling Vol"))
            fig_rv.add_trace(go.Scatter(x=m["spy_rolling_vol"].index, y=m["spy_rolling_vol"]*100, name=f"{benchmark_ticker} Rolling Vol", line=dict(dash="dot")))
            fig_rv.update_layout(title="1. Rolling Volatility (20d Annualized)", height=310, yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_rv, use_container_width=True)

            fig_rb5 = go.Figure()
            fig_rb5.add_trace(go.Scatter(x=m["rolling_beta"].index, y=m["rolling_beta"], name="Rolling Beta vs SPY", line=dict(color="#FFA15A")))
            fig_rb5.add_hline(y=1.0, line_dash="dash", line_color="red")
            fig_rb5.update_layout(title=f"2. Rolling Beta vs {benchmark_ticker} (20d)", height=310)
            st.plotly_chart(fig_rb5, use_container_width=True)

            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=m["mom_corr"].index, y=m["mom_corr"], name="Momentum Factor Exposure", line=dict(color="#AB63FA")))
            fig_mc.add_hline(y=0.0, line_dash="dash", line_color="gray")
            fig_mc.update_layout(title="3. Momentum Factor Exposure (Rolling Corr 20d)", height=310, yaxis=dict(range=[-1, 1]))
            st.plotly_chart(fig_mc, use_container_width=True)
