"""Bước 5 — Enhanced Risk Metrics: Rolling Beta vs SPY (myapp7.py)
Bổ sung chỉ số rủi ro nâng cao:
- Tính Rolling Beta (20 ngày) = Cov(Portfolio, SPY) / Var(SPY) để đo độ nhạy cảm của danh mục so với thị trường.
- Tính Rolling Alpha (CAPM adjusted return) - tính sẵn trong data pipeline.
- Hiển thị Latest Beta trên KPI và biểu đồ biến thiên Rolling Beta theo thời gian với đường tham chiếu Beta = 1.0.
- Diễn giải ý nghĩa tài chính tự động (đòn bẩy thị trường vs phòng hộ).
"""
from datetime import datetime, date
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as pxâsssxx

# 1. Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Enhanced Risk Metrics: Rolling Beta - Portfolio Terminal",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Portfolio Terminal — Enhanced Risk Metrics & Rolling Beta")
st.caption("Bước 5 trong lộ trình Portfolio Management App — Đo lường Độ nhạy Thị trường (Rolling Beta 20-Day)")

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
    beta_window = st.slider("Cửa sổ Rolling Beta (ngày giao dịch):", min_value=10, max_value=60, value=20, step=5)

    st.subheader("⚖️ Điều chỉnh Trọng số (Sliders)")
    raw_weights = {}
    default_w = 1.0 / len(tickers) if tickers else 1.0
    for t in tickers:
        raw_weights[t] = st.slider(f"Trọng số {t}:", min_value=0.0, max_value=1.0, value=float(default_w), step=0.05)

    total_raw = sum(raw_weights.values())
    if total_raw > 0:
        normalized_weights = {k: v / total_raw for k, v in raw_weights.items()}
    else:
        normalized_weights = {k: 1.0 / len(tickers) for k in tickers} if tickers else {}

    st.caption("📌 **Trọng số chuẩn hóa:**")
    for k, v in normalized_weights.items():
        st.write(f"- **{k}**: `{v * 100:.1f}%`")

    btn_reload = st.button("🚀 Cập nhật Terminal", use_container_width=True)


# 3. Hàm tải dữ liệu và tính toán
@st.cache_data(ttl=3600)
def fetch_terminal_data(assets: list, benchmark: str, start: str, end: str):
    if not assets:
        return None, "Danh sách tài sản rỗng."

    symbols_to_download = list(dict.fromkeys(assets + [benchmark]))
    try:
        raw_data = yf.download(symbols_to_download, start=start, end=end, auto_adjust=True)
    except Exception as e:
        return None, f"Lỗi tải dữ liệu: {e}"

    if raw_data.empty:
        return None, "Không tìm thấy dữ liệu."

    if "Close" in raw_data:
        close_df = raw_data["Close"].copy()
    else:
        close_df = raw_data.copy()

    if isinstance(close_df, pd.Series):
        close_df = close_df.to_frame(name=symbols_to_download[0])

    if isinstance(close_df.columns, pd.MultiIndex):
        close_df.columns = close_df.columns.get_level_values(-1)

    close_df.dropna(how="all", inplace=True)
    close_df.ffill(inplace=True)
    close_df.dropna(inplace=True)

    if benchmark not in close_df.columns:
        return None, f"Không tìm thấy benchmark {benchmark}."

    for a in assets:
        if a not in close_df.columns:
            return None, f"Không tìm thấy mã {a}."

    return close_df, None


def calculate_portfolio_with_beta(close_df: pd.DataFrame, assets: list, benchmark: str, weights_dict: dict, rf: float, window: int = 20):
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
    spy_sharpe = (spy_daily.mean() * 252 - rf) / spy_vol if spy_vol > 0 else 0.0

    port_wealth = (1 + port_daily).cumprod()
    port_dd = (port_wealth - port_wealth.cummax()) / port_wealth.cummax()
    spy_wealth = (1 + spy_daily).cumprod()
    spy_dd = (spy_wealth - spy_wealth.cummax()) / spy_wealth.cummax()

    # Tính Rolling Beta (20-day) = Cov(Portfolio, SPY) / Var(SPY)
    rolling_cov = port_daily.rolling(window).cov(spy_daily)
    rolling_var = spy_daily.rolling(window).var()
    rolling_beta = rolling_cov / rolling_var
    latest_beta = float(rolling_beta.dropna().iloc[-1]) if not rolling_beta.dropna().empty else 1.0

    # Tính Rolling Alpha (CAPM adjustment) - tính sẵn trong data pipeline
    rolling_alpha = (port_daily.rolling(window).mean() * 252) - rolling_beta * (spy_daily.rolling(window).mean() * 252)

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
        "spy_sharpe": spy_sharpe,
        "port_dd": port_dd,
        "spy_dd": spy_dd,
        "rolling_beta": rolling_beta,
        "latest_beta": latest_beta,
        "rolling_alpha": rolling_alpha,
        "signals": signals,
        "win_rate": win_rate,
        "daily_returns": daily_returns,
        "close_df": close_df,
    }


# 4. Hiển thị Giao diện
if not tickers:
    st.warning("⚠️ Vui lòng nhập danh sách mã cổ phiếu hợp lệ.")
else:
    with st.spinner("Đang xử lý dữ liệu và tính toán Rolling Beta..."):
        close_data, err = fetch_terminal_data(tickers, benchmark_ticker, str(start_date), str(end_date))

    if err:
        st.error(f"⚠️ {err}")
    else:
        m = calculate_portfolio_with_beta(close_data, tickers, benchmark_ticker, normalized_weights, rf_rate, beta_window)

        # 4 Tab
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 **Portfolio**",
            "🛡️ **Risk & Rolling Beta**",
            "📡 **Signals**",
            "🔬 **Backtest**"
        ])

        # ==================== TAB 1: PORTFOLIO ====================
        with tab1:
            st.subheader("📈 Tổng Quan Hiệu Suất Danh Mục vs SPY")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🌟 Portfolio Return", f"{m['port_total_return'] * 100:.2f}%", delta=f"{m['alpha'] * 100:.2f}% vs SPY")
            c2.metric("🏛️ SPY Return", f"{m['spy_total_return'] * 100:.2f}%")
            c3.metric("🔥 Excess Alpha", f"{m['alpha'] * 100:.2f}%")
            c4.metric("🎯 Sharpe Ratio", f"{m['port_sharpe']:.2f}")

            st.divider()
            col_pie, col_chart = st.columns([1, 2])
            with col_pie:
                st.subheader("⚖️ Phân Bổ Trọng Số")
                fig_alloc = px.pie(
                    names=list(normalized_weights.keys()),
                    values=[v * 100 for v in normalized_weights.values()],
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig_alloc.update_traces(textposition='inside', textinfo='percent+label')
                fig_alloc.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig_alloc, use_container_width=True)

            with col_chart:
                st.subheader(f"📈 Đường Tích Lũy: Danh mục vs {benchmark_ticker}")
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(x=m["port_cum"].index, y=m["port_cum"] * 100, mode="lines", name="Portfolio", line=dict(color="#00CC96", width=2.5)))
                fig_cum.add_trace(go.Scatter(x=m["spy_cum"].index, y=m["spy_cum"] * 100, mode="lines", name=benchmark_ticker, line=dict(color="#636EFA", width=1.8, dash="dash")))
                fig_cum.update_layout(height=380, xaxis_title="Thời gian", yaxis_title="Lợi nhuận tích lũy (%)", yaxis=dict(ticksuffix="%"))
                st.plotly_chart(fig_cum, use_container_width=True)

        # ==================== TAB 2: RISK & ROLLING BETA ====================
        with tab2:
            st.subheader("🛡️ Phân Tích Rủi Ro Nâng Cao & Rolling Beta (20 Ngày)")

            # KPI Beta & Nhận định tài chính
            b_col1, b_col2, b_col3 = st.columns(3)
            b_col1.metric("⚡ Latest Rolling Beta", f"{m['latest_beta']:.2f}")
            b_col2.metric("📉 Portfolio Volatility", f"{m['port_vol'] * 100:.2f}%")
            b_col3.metric("⚡ Max Drawdown", f"{m['port_dd'].min() * 100:.2f}%")

            # Hộp diễn giải ý nghĩa Beta
            if m['latest_beta'] > 1.0:
                excess_pct = (m['latest_beta'] - 1.0) * 100
                st.info(f"💡 **Nhận định Định lượng:** Beta mới nhất = **{m['latest_beta']:.2f}** → Danh mục **nhạy cảm hơn thị trường ~{excess_pct:.1f}%**, mang tính chất **“đòn bẩy” (leveraged exposure)** so với thị trường chứ không phải phòng hộ (hedge). Khi SPY tăng 1%, danh mục có xu hướng biến động ~{m['latest_beta']:.2f}%.")
            else:
                defensive_pct = (1.0 - m['latest_beta']) * 100
                st.info(f"💡 **Nhận định Định lượng:** Beta mới nhất = **{m['latest_beta']:.2f}** → Danh mục **ít nhạy cảm hơn thị trường ~{defensive_pct:.1f}%**, mang tính chất **phòng thủ (defensive/hedging)**.")

            st.divider()

            # Biểu đồ Rolling Beta theo thời gian
            st.subheader(f"📈 Biểu Đồ Rolling Beta ({beta_window} Ngày) vs {benchmark_ticker}")
            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(
                x=m["rolling_beta"].index,
                y=m["rolling_beta"],
                mode="lines",
                name=f"Rolling Beta ({beta_window}d)",
                line=dict(color="#FFA15A", width=2.5)
            ))
            # Đường tham chiếu Market Beta = 1.0
            fig_beta.add_hline(
                y=1.0,
                line_dash="dash",
                line_color="red",
                annotation_text="Market Beta = 1.0 (SPY)",
                annotation_position="bottom right"
            )
            fig_beta.update_layout(
                height=420,
                xaxis_title="Thời gian",
                yaxis_title="Beta Hệ Số",
                hovermode="x unified"
            )
            st.plotly_chart(fig_beta, use_container_width=True)

            # Drawdown & Ma trận tương quan
            col_dd, col_corr = st.columns(2)
            with col_dd:
                st.subheader("📉 Biểu Đồ Drawdown So Sánh")
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=m["port_dd"].index, y=m["port_dd"] * 100, mode="lines", name="Portfolio Drawdown", fill="tozeroy", line=dict(color="#EF553B")))
                fig_dd.add_trace(go.Scatter(x=m["spy_dd"].index, y=m["spy_dd"] * 100, mode="lines", name=f"{benchmark_ticker} Drawdown", line=dict(color="#AB63FA", dash="dot")))
                fig_dd.update_layout(height=350, yaxis=dict(ticksuffix="%"))
                st.plotly_chart(fig_dd, use_container_width=True)

            with col_corr:
                st.subheader("🔥 Ma Trận Tương Quan")
                corr_df = m["daily_returns"][tickers + [benchmark_ticker]].corr()
                fig_corr = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1.0, zmax=1.0)
                fig_corr.update_layout(height=350)
                st.plotly_chart(fig_corr, use_container_width=True)

            st.divider()
            st.subheader("📋 Bảng So Sánh Chỉ Số Biến Động")
            risk_summary = pd.DataFrame({
                "Chỉ số": ["Độ Biến Động (Volatility Năm)", "Mức Sụt Giảm Lớn Nhất (Max Drawdown)", "Sharpe Ratio", "Latest Rolling Beta"],
                "Portfolio": [f"{m['port_vol'] * 100:.2f}%", f"{m['port_dd'].min() * 100:.2f}%", f"{m['port_sharpe']:.2f}", f"{m['latest_beta']:.2f}"],
                f"Benchmark ({benchmark_ticker})": [f"{m['spy_vol'] * 100:.2f}%", f"{m['spy_dd'].min() * 100:.2f}%", f"{m['spy_sharpe']:.2f}", "1.00"]
            })
            st.dataframe(risk_summary.set_index("Chỉ số"), use_container_width=True)

        # ==================== TAB 3: SIGNALS ====================
        with tab3:
            st.subheader("📡 Tín Hiệu Giao Dịch MA5 / MA20")
            sig_cols = st.columns(len(tickers))
            for i, t in enumerate(tickers):
                info = m["signals"][t]
                with sig_cols[i]:
                    badge = "🟢 BUY" if info["Signal"] == "BUY" else "🔴 SELL"
                    st.metric(f"{t}", f"${info['Price']:.2f}", delta=badge)
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

        # ==================== TAB 4: BACKTEST ====================
        with tab4:
            st.subheader(f"🔬 Hiệu Suất Lịch Sử vs {benchmark_ticker}")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("🏆 Win Rate", f"{m['win_rate']:.1f}%")
            b2.metric("📅 Tổng Số Phiên Giao Dịch", f"{len(m['port_daily'])}")
            b3.metric("🔼 Lợi Nhuận Ngày Tốt Nhất", f"{m['port_daily'].max() * 100:.2f}%")
            b4.metric("🔽 Lợi Nhuận Ngày Kém Nhất", f"{m['port_daily'].min() * 100:.2f}%")

            st.divider()
            st.subheader("📊 Histogram Phân Phối Lợi Nhuận Ngày")
            compare_df = pd.DataFrame({"Portfolio": m["port_daily"] * 100, benchmark_ticker: m["spy_daily"] * 100})
            fig_hist = px.histogram(compare_df, barmode="overlay", nbins=60, opacity=0.6, color_discrete_map={"Portfolio": "#00CC96", benchmark_ticker: "#636EFA"})
            fig_hist.update_layout(height=380, xaxis_title="Lợi nhuận ngày (%)")
            st.plotly_chart(fig_hist, use_container_width=True)
