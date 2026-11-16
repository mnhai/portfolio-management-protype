"""Bước 6 — Momentum Exposure Factor Panel (myapp8.py)
Xây dựng Synthetic Momentum Factor từ SPY:
- Định nghĩa Momentum = P(t) / P(t-12) - 1 (12-day lookback).
- Kỹ thuật chuẩn hedge fund: Tạo yếu tố đà tăng trưởng tổng hợp (synthetic momentum factor) khi không có bộ dữ liệu factor chuẩn.
- Đo lường tương quan rolling (20 ngày) giữa danh mục và yếu tố momentum.
- Bổ sung Tab 5: "Factor Panel" hiển thị mức độ phơi nhiễm (Exposure) với đà thị trường.
"""
from datetime import datetime, date
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

# 1. Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Momentum Exposure Factor Panel - Portfolio Terminal",
    page_icon="🌪️",
    layout="wide"
)

st.title("🌪️ Portfolio Terminal — Momentum Factor Exposure Panel")
st.caption("Bước 6 trong lộ trình Portfolio Management App — Phân tích Nhân tố Momentum Tổng hợp (Synthetic Factor)")

# 2. Thanh bên (Sidebar)
with st.sidebar:
    st.header("⚙️ Cấu hình Danh mục")
    default_tickers = "AAPL, MSFT, NVDA"
    tickers_input = st.text_input("Mã cổ phiếu danh mục:", default_tickers)
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    benchmark_ticker = st.text_input("Benchmark thị trường:", "SPY").strip().upper()

    start_date = st.date_input("Ngày bắt đầu:", date(2023, 1, 1))
    end_date = st.date_input("Ngày kết thúc:", datetime.today().date())
    rf_rate = st.number_input("Lãi suất phi rủi ro:", value=0.02, step=0.005, format="%.3f")
    mom_lookback = st.slider("Lookback Momentum (phiên):", min_value=5, max_value=30, value=12, step=1)

    st.subheader("⚖️ Điều chỉnh Trọng số")
    raw_weights = {}
    default_w = 1.0 / len(tickers) if tickers else 1.0
    for t in tickers:
        raw_weights[t] = st.slider(f"Trọng số {t}:", 0.0, 1.0, float(default_w), 0.05)

    total_raw = sum(raw_weights.values())
    normalized_weights = {k: v / total_raw for k, v in raw_weights.items()} if total_raw > 0 else {k: 1.0 / len(tickers) for k in tickers}

    st.caption("📌 **Trọng số chuẩn hóa:**")
    for k, v in normalized_weights.items():
        st.write(f"- **{k}**: `{v * 100:.1f}%`")

    btn_reload = st.button("🚀 Cập nhật Terminal", use_container_width=True)


# 3. Hàm tải dữ liệu và tính toán Factor
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
        return None, f"Thiếu mã benchmark {benchmark}."

    for a in assets:
        if a not in close_df.columns:
            return None, f"Thiếu dữ liệu mã {a}."

    return close_df, None


def calculate_portfolio_with_factors(close_df: pd.DataFrame, assets: list, benchmark: str, weights_dict: dict, rf: float, lookback: int = 12):
    daily_returns = close_df.pct_change().dropna()
    weights_vec = np.array([weights_dict[a] for a in assets])

    port_daily = daily_returns[assets].dot(weights_vec)
    spy_daily = daily_returns[benchmark]

    port_cum = (1 + port_daily).cumprod() - 1
    spy_cum = (1 + spy_daily).cumprod() - 1

    port_total_return = (1 + port_daily).cumprod().iloc[-1] - 1
    spy_total_return = (1 + spy_daily).cumprod().iloc[-1] - 1
    alpha = port_total_return - spy_total_return

    port_vol = port_daily.std() * np.sqrt(252)
    spy_vol = spy_daily.std() * np.sqrt(252)
    port_sharpe = (port_daily.mean() * 252 - rf) / port_vol if port_vol > 0 else 0.0

    port_wealth = (1 + port_daily).cumprod()
    port_dd = (port_wealth - port_wealth.cummax()) / port_wealth.cummax()
    spy_wealth = (1 + spy_daily).cumprod()
    spy_dd = (spy_wealth - spy_wealth.cummax()) / spy_wealth.cummax()

    # Rolling Beta vs SPY (20d)
    rolling_beta = port_daily.rolling(20).cov(spy_daily) / spy_daily.rolling(20).var()
    latest_beta = float(rolling_beta.dropna().iloc[-1]) if not rolling_beta.dropna().empty else 1.0

    # ==================== SYNTHETIC MOMENTUM FACTOR ====================
    # P(t) / P(t - 12) - 1
    spy_price = close_df[benchmark]
    synthetic_mom_factor = (spy_price / spy_price.shift(lookback)) - 1
    portfolio_mom = (port_wealth / port_wealth.shift(lookback)) - 1

    # Rolling Correlation (20d) giữa Portfolio Return và Synthetic Momentum Factor
    mom_aligned_df = pd.DataFrame({
        "Portfolio_Return": port_daily,
        "Portfolio_Mom": portfolio_mom,
        "Synthetic_Mom": synthetic_mom_factor
    }).dropna()

    rolling_mom_corr = mom_aligned_df["Portfolio_Return"].rolling(20).corr(mom_aligned_df["Synthetic_Mom"])
    latest_mom_corr = float(rolling_mom_corr.dropna().iloc[-1]) if not rolling_mom_corr.dropna().empty else 0.0

    # Factor Beta: Cov(Rp, Mom_Factor) / Var(Mom_Factor)
    rolling_factor_beta = mom_aligned_df["Portfolio_Return"].rolling(20).cov(mom_aligned_df["Synthetic_Mom"]) / mom_aligned_df["Synthetic_Mom"].rolling(20).var()
    latest_factor_beta = float(rolling_factor_beta.dropna().iloc[-1]) if not rolling_factor_beta.dropna().empty else 0.0

    # Signals
    signals = {}
    for a in assets:
        ma5 = close_df[a].rolling(5).mean()
        ma20 = close_df[a].rolling(20).mean()
        sig = "BUY" if ma5.iloc[-1] > ma20.iloc[-1] else "SELL"
        signals[a] = {"Price": close_df[a].iloc[-1], "MA5": ma5.iloc[-1], "MA20": ma20.iloc[-1], "Signal": sig}

    win_rate = (port_daily > 0).mean() * 100

    return {
        "port_daily": port_daily,
        "spy_daily": spy_daily,
        "port_cum": port_cum,
        "spy_cum": spy_cum,
        "port_total_return": port_total_return,
        "spy_total_return": spy_total_return,
        "alpha": alpha,
        "port_vol": port_vol,
        "spy_vol": spy_vol,
        "port_sharpe": port_sharpe,
        "port_dd": port_dd,
        "spy_dd": spy_dd,
        "rolling_beta": rolling_beta,
        "latest_beta": latest_beta,
        "synthetic_mom_factor": synthetic_mom_factor,
        "portfolio_mom": portfolio_mom,
        "rolling_mom_corr": rolling_mom_corr,
        "latest_mom_corr": latest_mom_corr,
        "rolling_factor_beta": rolling_factor_beta,
        "latest_factor_beta": latest_factor_beta,
        "signals": signals,
        "win_rate": win_rate,
        "daily_returns": daily_returns,
        "close_df": close_df,
    }


# 4. Hiển thị Giao diện
if not tickers:
    st.warning("⚠️ Vui lòng nhập danh sách mã hợp lệ.")
else:
    with st.spinner("Đang tính toán Synthetic Momentum Factor..."):
        close_data, err = fetch_terminal_data(tickers, benchmark_ticker, str(start_date), str(end_date))

    if err:
        st.error(f"⚠️ {err}")
    else:
        m = calculate_portfolio_with_factors(close_data, tickers, benchmark_ticker, normalized_weights, rf_rate, mom_lookback)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 **Portfolio**",
            "🛡️ **Risk & Beta**",
            "📡 **Signals**",
            "🔬 **Backtest**",
            "🌪️ **Momentum Factors**"
        ])

        # TAB 1: PORTFOLIO
        with tab1:
            st.subheader("📈 Hiệu Suất Tổng Thể")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🌟 Portfolio Return", f"{m['port_total_return'] * 100:.2f}%", delta=f"{m['alpha'] * 100:.2f}% vs SPY")
            c2.metric("🏛️ SPY Return", f"{m['spy_total_return'] * 100:.2f}%")
            c3.metric("🔥 Excess Alpha", f"{m['alpha'] * 100:.2f}%")
            c4.metric("🎯 Sharpe Ratio", f"{m['port_sharpe']:.2f}")

            st.divider()
            col_pie, col_chart = st.columns([1, 2])
            with col_pie:
                st.subheader("⚖️ Trọng Số Phân Bổ")
                fig_alloc = px.pie(names=list(normalized_weights.keys()), values=[v * 100 for v in normalized_weights.values()], hole=0.45)
                fig_alloc.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig_alloc, use_container_width=True)
            with col_chart:
                st.subheader("📈 Lũy Kế Danh Mục vs SPY")
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(x=m["port_cum"].index, y=m["port_cum"] * 100, mode="lines", name="Portfolio", line=dict(color="#00CC96", width=2.5)))
                fig_cum.add_trace(go.Scatter(x=m["spy_cum"].index, y=m["spy_cum"] * 100, mode="lines", name=benchmark_ticker, line=dict(color="#636EFA", width=1.8, dash="dash")))
                fig_cum.update_layout(height=380, yaxis=dict(ticksuffix="%"))
                st.plotly_chart(fig_cum, use_container_width=True)

        # TAB 2: RISK
        with tab2:
            st.subheader("🛡️ Quản Trị Rủi Ro & Beta")
            r1, r2, r3 = st.columns(3)
            r1.metric("⚡ Latest Rolling Beta", f"{m['latest_beta']:.2f}")
            r2.metric("📉 Volatility", f"{m['port_vol'] * 100:.2f}%")
            r3.metric("⚡ Max Drawdown", f"{m['port_dd'].min() * 100:.2f}%")

            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(x=m["rolling_beta"].index, y=m["rolling_beta"], name="Rolling Beta (20d)", line=dict(color="#FFA15A", width=2)))
            fig_beta.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Market Beta = 1.0")
            fig_beta.update_layout(height=350, xaxis_title="Thời gian", yaxis_title="Beta")
            st.plotly_chart(fig_beta, use_container_width=True)

        # TAB 3: SIGNALS
        with tab3:
            st.subheader("📡 Tín Hiệu Kỹ Thuật MA5/MA20")
            s_cols = st.columns(len(tickers))
            for i, t in enumerate(tickers):
                info = m["signals"][t]
                with s_cols[i]:
                    st.metric(t, f"${info['Price']:.2f}", delta="🟢 BUY" if info["Signal"] == "BUY" else "🔴 SELL")

        # TAB 4: BACKTEST
        with tab4:
            st.subheader("🔬 Thống Kê Lịch Sử")
            st.metric("🏆 Win Rate (% Ngày Dương)", f"{m['win_rate']:.1f}%")

        # ==================== TAB 5: MOMENTUM FACTORS ====================
        with tab5:
            st.subheader("🌪️ Phân Tích Phơi Nhiễm Yếu Tố Momentum (Synthetic Factor)")
            
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("🌀 Latest Momentum Corr", f"{m['latest_mom_corr']:.2f}")
            f2.metric("⚡ Factor Beta (Momentum)", f"{m['latest_factor_beta']:.2f}")
            
            latest_port_mom = float(m["portfolio_mom"].dropna().iloc[-1]) if not m["portfolio_mom"].dropna().empty else 0.0
            latest_spy_mom = float(m["synthetic_mom_factor"].dropna().iloc[-1]) if not m["synthetic_mom_factor"].dropna().empty else 0.0
            f3.metric(f"🚀 Portfolio {mom_lookback}d Momentum", f"{latest_port_mom * 100:.2f}%")
            f4.metric(f"🏛️ SPY {mom_lookback}d Momentum", f"{latest_spy_mom * 100:.2f}%")

            # Nhận định chiến lược Momentum Tilt
            if m["latest_mom_corr"] > 0.3:
                tilt_type = "MOMENTUM RIDER (Bắt đà xu hướng)"
                desc = "Danh mục đang có tương quan dương mạnh với đà thị trường, hưởng lợi lớn trong giai đoạn xu hướng tăng mạnh nhưng dễ chịu rủi ro khi thị trường đảo chiều đột ngột."
            elif m["latest_mom_corr"] < -0.3:
                tilt_type = "CONTRARIAN (Đi ngược đà)"
                desc = "Danh mục có tương quan âm với đà thị trường, mang tính chất đón sóng phục hồi hoặc phòng thủ nghịch chu kỳ."
            else:
                tilt_type = "MOMENTUM NEUTRAL (Trung lập với đà)"
                desc = "Lợi nhuận danh mục không phụ thuộc đáng kể vào đà ngắn hạn của SPY, thể hiện tính độc lập cao."

            st.info(f"💡 **Phân Loại Phong Cách:** **{tilt_type}**\n\n{desc}")

            st.divider()

            # Biểu đồ 1: Rolling Correlation với Synthetic Momentum
            st.subheader(f"📈 Tương Quan Rolling (20 Ngày) giữa Danh Mục & Synthetic Momentum Factor")
            fig_mom_corr = go.Figure()
            fig_mom_corr.add_trace(go.Scatter(
                x=m["rolling_mom_corr"].index,
                y=m["rolling_mom_corr"],
                mode="lines",
                name="Rolling Momentum Correlation",
                line=dict(color="#AB63FA", width=2.2)
            ))
            fig_mom_corr.add_hline(y=0.0, line_dash="dash", line_color="gray")
            fig_mom_corr.update_layout(height=380, xaxis_title="Thời gian", yaxis_title="Hệ số tương quan (-1 đến 1)", yaxis=dict(range=[-1, 1]))
            st.plotly_chart(fig_mom_corr, use_container_width=True)

            # Biểu đồ 2: So sánh Momentum 12 ngày giữa Portfolio vs SPY
            st.subheader(f"📊 So Sánh Đà Tăng Trưởng ({mom_lookback}-Day Momentum): Portfolio vs SPY")
            fig_mom_comp = go.Figure()
            fig_mom_comp.add_trace(go.Scatter(
                x=m["portfolio_mom"].index,
                y=m["portfolio_mom"] * 100,
                mode="lines",
                name="Portfolio Momentum",
                line=dict(color="#00CC96", width=2)
            ))
            fig_mom_comp.add_trace(go.Scatter(
                x=m["synthetic_mom_factor"].index,
                y=m["synthetic_mom_factor"] * 100,
                mode="lines",
                name="SPY Synthetic Momentum",
                line=dict(color="#EF553B", width=1.5, dash="dot")
            ))
            fig_mom_comp.update_layout(height=380, xaxis_title="Thời gian", yaxis_title=f"Lợi nhuận {mom_lookback} ngày (%)", yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_mom_comp, use_container_width=True)
