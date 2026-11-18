"""Bước 7 — Max Sharpe Portfolio Optimization Engine (myapp9.py)
Dùng scipy.optimize.minimize (thuật toán SLSQP) để tối ưu hóa trọng số danh mục:
- Mục tiêu: Tối đa hóa Sharpe Ratio (tương đương cực tiểu hóa -Sharpe Ratio).
- Ràng buộc thực tế:
  * Không bán khống: w_i >= 0.
  * Tổng trọng số: sum(w_i) = 1.0.
  * Giới hạn tỷ trọng tối đa cho mỗi tài sản (max weight per asset).
- Hiển thị danh mục tối ưu, lợi nhuận kỳ vọng, độ biến động và Sharpe Ratio tối ưu.
- So sánh hiệu suất lũy kế 3 đường: User Weights vs Max Sharpe Optimized vs Benchmark (SPY).
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
    page_title="Max Sharpe Portfolio Optimization Engine",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Portfolio Terminal — Max Sharpe Optimization Engine")
st.caption("Bước 7 trong lộ trình Portfolio Management App — Tối ưu hóa Trọng số Hiện đại theo Markowitz & Sharpe")

# 2. Thanh bên (Sidebar)
with st.sidebar:
    st.header("⚙️ Cấu hình Danh mục")
    default_tickers = "AAPL, MSFT, NVDA"
    tickers_input = st.text_input("Mã cổ phiếu danh mục:", default_tickers)
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    benchmark_ticker = st.text_input("Benchmark thị trường:", "SPY").strip().upper()

    start_date = st.date_input("Ngày bắt đầu:", date(2023, 1, 1))
    end_date = st.date_input("Ngày kết thúc:", datetime.today().date())
    rf_rate = st.number_input("Lãi suất phi rủi ro (Risk-Free Rate):", value=0.02, step=0.005, format="%.3f")

    st.subheader("🛠️ Ràng Buộc Tối Ưu Hóa (Optimization Constraints)")
    max_weight_limit = st.slider("Giới hạn tỷ trọng tối đa mỗi mã (Max Weight):", min_value=0.2, max_value=1.0, value=0.6, step=0.05)

    st.subheader("⚖️ Trọng Số Tự Chọn (User Weights)")
    raw_weights = {}
    default_w = 1.0 / len(tickers) if tickers else 1.0
    for t in tickers:
        raw_weights[t] = st.slider(f"Trọng số {t}:", 0.0, 1.0, float(default_w), 0.05)

    total_raw = sum(raw_weights.values())
    normalized_weights = {k: v / total_raw for k, v in raw_weights.items()} if total_raw > 0 else {k: 1.0 / len(tickers) for k in tickers}

    btn_reload = st.button("🚀 Chạy Tối Ưu Hóa & Cập Nhật", use_container_width=True)


# 3. Hàm tải dữ liệu và tối ưu hóa
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
    """
    Tối ưu hóa tìm bộ trọng số tối đa hóa Sharpe Ratio bằng SLSQP.
    """
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
        if port_vol == 0:
            return 0
        return -(port_ret - rf) / port_vol

    res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)

    opt_weights = res.x if res.success else w0
    opt_ret = np.dot(opt_weights, mean_ret)
    opt_vol = np.sqrt(np.dot(opt_weights.T, np.dot(cov_mat, opt_weights)))
    opt_sharpe = (opt_ret - rf) / opt_vol if opt_vol > 0 else 0.0

    return {
        "success": res.success,
        "weights": dict(zip(assets, opt_weights)),
        "expected_return": opt_ret,
        "volatility": opt_vol,
        "sharpe_ratio": opt_sharpe,
    }


def calculate_full_portfolio(close_df: pd.DataFrame, assets: list, benchmark: str, user_weights: dict, rf: float, max_weight: float):
    daily_returns = close_df.pct_change().dropna()
    weights_vec = np.array([user_weights[a] for a in assets])

    # User Portfolio
    user_daily = daily_returns[assets].dot(weights_vec)
    user_cum = (1 + user_daily).cumprod() - 1
    user_vol = user_daily.std() * np.sqrt(252)
    user_sharpe = (user_daily.mean() * 252 - rf) / user_vol if user_vol > 0 else 0.0
    user_total = user_cum.iloc[-1]

    # Benchmark SPY
    spy_daily = daily_returns[benchmark]
    spy_cum = (1 + spy_daily).cumprod() - 1
    spy_vol = spy_daily.std() * np.sqrt(252)
    spy_sharpe = (spy_daily.mean() * 252 - rf) / spy_vol if spy_vol > 0 else 0.0
    spy_total = spy_cum.iloc[-1]

    # Optimize Max Sharpe Portfolio
    opt_result = optimize_max_sharpe(daily_returns, assets, rf, max_weight)
    opt_vec = np.array([opt_result["weights"][a] for a in assets])
    opt_daily = daily_returns[assets].dot(opt_vec)
    opt_cum = (1 + opt_daily).cumprod() - 1
    opt_total = opt_cum.iloc[-1]

    # Drawdowns
    user_wealth = (1 + user_daily).cumprod()
    user_dd = (user_wealth - user_wealth.cummax()) / user_wealth.cummax()

    opt_wealth = (1 + opt_daily).cumprod()
    opt_dd = (opt_wealth - opt_wealth.cummax()) / opt_wealth.cummax()

    spy_wealth = (1 + spy_daily).cumprod()
    spy_dd = (spy_wealth - spy_wealth.cummax()) / spy_wealth.cummax()

    # Rolling Volatility (20d annualized)
    rolling_vol = user_daily.rolling(20).std() * np.sqrt(252)
    spy_rolling_vol = spy_daily.rolling(20).std() * np.sqrt(252)

    # Rolling Beta vs SPY
    rolling_beta = user_daily.rolling(20).cov(spy_daily) / spy_daily.rolling(20).var()
    latest_beta = float(rolling_beta.dropna().iloc[-1]) if not rolling_beta.dropna().empty else 1.0

    # Synthetic Momentum Factor
    spy_price = close_df[benchmark]
    synthetic_mom = (spy_price / spy_price.shift(12)) - 1
    port_mom = (user_wealth / user_wealth.shift(12)) - 1
    rolling_mom_corr = user_daily.rolling(20).corr(synthetic_mom)
    latest_mom_corr = float(rolling_mom_corr.dropna().iloc[-1]) if not rolling_mom_corr.dropna().empty else 0.0
    win_rate = (user_daily > 0).mean() * 100

    # Signals
    signals = {}
    for a in assets:
        ma5 = close_df[a].rolling(5).mean()
        ma20 = close_df[a].rolling(20).mean()
        sig = "BUY" if ma5.iloc[-1] > ma20.iloc[-1] else "SELL"
        signals[a] = {"Price": close_df[a].iloc[-1], "MA5": ma5.iloc[-1], "MA20": ma20.iloc[-1], "Signal": sig}

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
        "rolling_vol": rolling_vol,
        "spy_rolling_vol": spy_rolling_vol,
        "win_rate": win_rate,
        "opt_result": opt_result,
        "opt_daily": opt_daily,
        "opt_cum": opt_cum,
        "opt_total": opt_total,
        "opt_dd": opt_dd,
        "rolling_beta": rolling_beta,
        "latest_beta": latest_beta,
        "synthetic_mom": synthetic_mom,
        "port_mom": port_mom,
        "rolling_mom_corr": rolling_mom_corr,
        "latest_mom_corr": latest_mom_corr,
        "signals": signals,
        "daily_returns": daily_returns,
        "close_df": close_df,
    }


# 4. Hiển thị Giao diện
if not tickers:
    st.warning("⚠️ Vui lòng nhập danh sách mã hợp lệ.")
else:
    with st.spinner("Đang chạy cỗ máy tối ưu hóa Max Sharpe..."):
        close_data, err = fetch_terminal_data(tickers, benchmark_ticker, str(start_date), str(end_date))

    if err:
        st.error(f"⚠️ {err}")
    else:
        m = calculate_full_portfolio(close_data, tickers, benchmark_ticker, normalized_weights, rf_rate, max_weight_limit)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 **Optimization & Portfolio**",
            "🛡️ **Risk & Beta**",
            "📡 **Signals**",
            "🔬 **Backtest**",
            "🌪️ **Momentum Factors**"
        ])

        # TAB 1: OPTIMIZATION & PORTFOLIO
        with tab1:
            st.subheader("🎯 Cỗ Máy Tối Ưu Hóa Danh Mục (Max Sharpe Optimization)")
            opt = m["opt_result"]

            # So sánh 3 danh mục
            col_u, col_opt, col_spy = st.columns(3)
            with col_u:
                st.info("👤 **User Defined Portfolio**")
                st.metric("Total Return", f"{m['user_total'] * 100:.2f}%")
                st.metric("Volatility (Năm)", f"{m['user_vol'] * 100:.2f}%")
                st.metric("Sharpe Ratio", f"{m['user_sharpe']:.2f}")

            with col_opt:
                st.success("🌟 **Max Sharpe Optimized**")
                st.metric("Expected Return", f"{opt['expected_return'] * 100:.2f}%", delta=f"{(opt['expected_return'] - m['user_vol']) * 100:.2f}%")
                st.metric("Volatility (Tối ưu)", f"{opt['volatility'] * 100:.2f}%")
                st.metric("Optimized Sharpe", f"{opt['sharpe_ratio']:.2f}", delta=f"{opt['sharpe_ratio'] - m['user_sharpe']:.2f}")

            with col_spy:
                st.warning(f"🏛️ **Benchmark ({benchmark_ticker})**")
                st.metric("SPY Return", f"{m['spy_total'] * 100:.2f}%")
                st.metric("SPY Volatility", f"{m['spy_vol'] * 100:.2f}%")
                st.metric("SPY Sharpe", f"{m['spy_sharpe']:.2f}")

            st.divider()

            # So sánh phân bổ trọng số
            col_w_bar, col_cum_chart = st.columns([1, 2])
            with col_w_bar:
                st.subheader("⚖️ So Sánh Trọng Số Phân Bổ")
                weights_compare_df = pd.DataFrame({
                    "User Weights": [normalized_weights[t] * 100 for t in tickers],
                    "Max Sharpe Weights": [opt["weights"][t] * 100 for t in tickers]
                }, index=tickers)
                fig_bar = px.bar(
                    weights_compare_df,
                    barmode="group",
                    labels={"value": "Tỷ trọng (%)", "index": "Mã cổ phiếu", "variable": "Danh mục"},
                    color_discrete_map={"User Weights": "#636EFA", "Max Sharpe Weights": "#00CC96"}
                )
                fig_bar.update_layout(height=400, yaxis=dict(ticksuffix="%"))
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_cum_chart:
                st.subheader("📈 So Sánh Đường Lũy Kế 3 Danh Mục")
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(x=m["opt_cum"].index, y=m["opt_cum"] * 100, mode="lines", name="🌟 MAX SHARPE OPTIMIZED", line=dict(color="#00CC96", width=3)))
                fig_cum.add_trace(go.Scatter(x=m["user_cum"].index, y=m["user_cum"] * 100, mode="lines", name="👤 USER WEIGHTS", line=dict(color="#636EFA", width=2)))
                fig_cum.add_trace(go.Scatter(x=m["spy_cum"].index, y=m["spy_cum"] * 100, mode="lines", name=f"🏛️ {benchmark_ticker}", line=dict(color="#FFA15A", width=1.5, dash="dash")))
                fig_cum.update_layout(height=400, xaxis_title="Thời gian", yaxis_title="Lợi nhuận tích lũy (%)", yaxis=dict(ticksuffix="%"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_cum, use_container_width=True)

        # TAB 2: RISK
        with tab2:
            st.subheader("🛡️ Quản Trị Rủi Ro & Rolling Beta (20 Ngày)")
            r1, r2, r3 = st.columns(3)
            r1.metric("⚡ Latest Rolling Beta", f"{m['latest_beta']:.2f}")
            r2.metric("📉 User Volatility", f"{m['user_vol'] * 100:.2f}%")
            r3.metric("📉 Optimized Volatility", f"{opt['volatility'] * 100:.2f}%")

            if m['latest_beta'] > 1.0:
                excess_pct = (m['latest_beta'] - 1.0) * 100
                st.info(f"💡 **Nhận định Định lượng:** Beta mới nhất = **{m['latest_beta']:.2f}** → Danh mục **nhạy cảm hơn thị trường ~{excess_pct:.1f}%**, mang tính chất **“đòn bẩy” (leveraged exposure)**.")
            else:
                defensive_pct = (1.0 - m['latest_beta']) * 100
                st.info(f"💡 **Nhận định Định lượng:** Beta mới nhất = **{m['latest_beta']:.2f}** → Danh mục **ít nhạy cảm hơn thị trường ~{defensive_pct:.1f}%**, mang tính **phòng thủ**.")

            st.subheader(f"📈 Biểu Đồ Rolling Beta (20 Ngày) vs {benchmark_ticker}")
            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(x=m["rolling_beta"].index, y=m["rolling_beta"], name="Rolling Beta (20d)", line=dict(color="#FFA15A", width=2)))
            fig_beta.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Market Beta = 1.0")
            fig_beta.update_layout(height=350, xaxis_title="Thời gian", yaxis_title="Beta", hovermode="x unified")
            st.plotly_chart(fig_beta, use_container_width=True)

            col_dd, col_corr = st.columns(2)
            with col_dd:
                st.subheader("📉 Biểu Đồ Drawdown So Sánh")
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=m["opt_dd"].index, y=m["opt_dd"] * 100, name="Optimized Drawdown", fill="tozeroy", line=dict(color="#00CC96")))
                fig_dd.add_trace(go.Scatter(x=m["user_dd"].index, y=m["user_dd"] * 100, name="User Drawdown", line=dict(color="#EF553B", dash="dot")))
                fig_dd.add_trace(go.Scatter(x=m["spy_dd"].index, y=m["spy_dd"] * 100, name=f"{benchmark_ticker} Drawdown", line=dict(color="#AB63FA", dash="dot")))
                fig_dd.update_layout(height=350, yaxis=dict(ticksuffix="%"))
                st.plotly_chart(fig_dd, use_container_width=True)
            with col_corr:
                st.subheader("🔥 Ma Trận Tương Quan")
                corr_df = m["daily_returns"][tickers + [benchmark_ticker]].corr()
                fig_corr = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1.0, zmax=1.0)
                fig_corr.update_layout(height=350)
                st.plotly_chart(fig_corr, use_container_width=True)

        # TAB 3: SIGNALS
        with tab3:
            st.subheader("📡 Tín Hiệu Giao Dịch MA5 / MA20")
            s_cols = st.columns(len(tickers))
            for i, t in enumerate(tickers):
                info = m["signals"][t]
                with s_cols[i]:
                    st.metric(t, f"${info['Price']:.2f}", delta="🟢 BUY" if info["Signal"] == "BUY" else "🔴 SELL")
                    st.write(f"- MA5: **${info['MA5']:.2f}**")
                    st.write(f"- MA20: **${info['MA20']:.2f}**")

            st.divider()
            st.subheader("📊 Biểu Đồ Xu Hướng Giá Kèm Tín Hiệu Cắt MA5/MA20")
            selected_ticker = st.selectbox("Chọn mã để xem chi tiết biểu đồ tín hiệu:", tickers)
            fig_sig = go.Figure()
            fig_sig.add_trace(go.Scatter(x=close_data.index, y=close_data[selected_ticker], name="Close Price", line=dict(color="#1f77b4")))
            fig_sig.add_trace(go.Scatter(x=close_data.index, y=close_data[selected_ticker].rolling(5).mean(), name="MA5", line=dict(color="#ff7f0e", dash="dot")))
            fig_sig.add_trace(go.Scatter(x=close_data.index, y=close_data[selected_ticker].rolling(20).mean(), name="MA20", line=dict(color="#2ca02c")))
            fig_sig.update_layout(height=400, xaxis_title="Thời gian", yaxis_title="Giá (USD)", hovermode="x unified")
            st.plotly_chart(fig_sig, use_container_width=True)

        # TAB 4: BACKTEST
        with tab4:
            st.subheader(f"🔬 Hiệu Suất Lịch Sử Chi Tiết vs {benchmark_ticker}")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("🏆 Win Rate (% Phiên Dương)", f"{m['win_rate']:.1f}%")
            b2.metric("📅 Tổng Số Phiên Giao Dịch", f"{len(m['user_daily'])}")
            b3.metric("🔼 Lợi Nhuận Ngày Tốt Nhất", f"{m['user_daily'].max() * 100:.2f}%")
            b4.metric("🔽 Lợi Nhuận Ngày Kém Nhất", f"{m['user_daily'].min() * 100:.2f}%")

            b_df = pd.DataFrame({
                "Chỉ số": ["Total Return", "Volatility (Năm)", "Sharpe Ratio", "Max Drawdown"],
                "User Portfolio": [f"{m['user_total']*100:.2f}%", f"{m['user_vol']*100:.2f}%", f"{m['user_sharpe']:.2f}", f"{m['user_dd'].min()*100:.2f}%"],
                "Max Sharpe Optimized": [f"{m['opt_total']*100:.2f}%", f"{opt['volatility']*100:.2f}%", f"{opt['sharpe_ratio']:.2f}", f"{m['opt_dd'].min()*100:.2f}%"],
                f"{benchmark_ticker} Benchmark": [f"{m['spy_total']*100:.2f}%", f"{m['spy_vol']*100:.2f}%", f"{m['spy_sharpe']:.2f}", f"{m['spy_dd'].min()*100:.2f}%"]
            })
            st.dataframe(b_df.set_index("Chỉ số"), use_container_width=True)

            st.subheader("📊 Histogram Phân Phối Lợi Nhuận Ngày")
            compare_df = pd.DataFrame({"User Portfolio": m["user_daily"] * 100, "Max Sharpe": m["opt_daily"] * 100, benchmark_ticker: m["spy_daily"] * 100})
            fig_hist = px.histogram(compare_df, barmode="overlay", nbins=60, opacity=0.55)
            fig_hist.update_layout(height=360, xaxis_title="Lợi nhuận ngày (%)", yaxis_title="Tần suất")
            st.plotly_chart(fig_hist, use_container_width=True)

        # TAB 5: MOMENTUM
        with tab5:
            st.subheader("🌪️ Phơi Nhiễm Yếu Tố Momentum (3 Biểu Đồ Phân Tích Chuyên Sâu)")
            st.metric("🌀 Latest Momentum Correlation", f"{m['latest_mom_corr']:.2f}")

            st.subheader("📉 1. Rolling Volatility (20 Ngày)")
            fig_rvol = go.Figure()
            fig_rvol.add_trace(go.Scatter(x=m["rolling_vol"].index, y=m["rolling_vol"] * 100, name="Portfolio Rolling Vol", line=dict(color="#EF553B")))
            fig_rvol.add_trace(go.Scatter(x=m["spy_rolling_vol"].index, y=m["spy_rolling_vol"] * 100, name=f"{benchmark_ticker} Rolling Vol", line=dict(color="#636EFA", dash="dot")))
            fig_rvol.update_layout(height=330, yaxis=dict(ticksuffix="%"), hovermode="x unified")
            st.plotly_chart(fig_rvol, use_container_width=True)

            st.subheader(f"⚡ 2. Rolling Beta vs {benchmark_ticker}")
            fig_rb5 = go.Figure()
            fig_rb5.add_trace(go.Scatter(x=m["rolling_beta"].index, y=m["rolling_beta"], name="Rolling Beta", line=dict(color="#FFA15A")))
            fig_rb5.add_hline(y=1.0, line_dash="dash", line_color="red")
            fig_rb5.update_layout(height=330, hovermode="x unified")
            st.plotly_chart(fig_rb5, use_container_width=True)

            st.subheader("🌪️ 3. Momentum Factor Exposure (Tương Quan Rolling 20 Ngày)")
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=m["rolling_mom_corr"].index, y=m["rolling_mom_corr"], name="Rolling Momentum Corr", line=dict(color="#AB63FA")))
            fig_mc.add_hline(y=0.0, line_dash="dash", line_color="gray")
            fig_mc.update_layout(height=330, yaxis=dict(range=[-1, 1]), hovermode="x unified")
            st.plotly_chart(fig_mc, use_container_width=True)
