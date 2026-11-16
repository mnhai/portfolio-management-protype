"""Bước 4 — Interactive Portfolio App with User-Defined Weights (myapp5.py)
Mini Bloomberg Terminal với 4 Tab:
- Tab 1 (Portfolio): Lợi nhuận danh mục, Lợi nhuận SPY, Alpha, Sharpe, biểu đồ Pie/Donut, đường tích lũy so với SPY.
- Tab 2 (Risk): Ma trận tương quan (Heatmap), so sánh Drawdown/Volatility với SPY.
- Tab 3 (Signals): Tín hiệu BUY/SELL theo MA5/MA20 cho từng mã cổ phiếu trong danh mục.
- Tab 4 (Backtest): Hiệu suất lịch sử chi tiết so với SPY, histogram so sánh lợi nhuận ngày, tỷ lệ ngày thắng (Win Rate).
Người dùng có thể tùy chỉnh trọng số từng tài sản linh hoạt qua thanh trượt (slider).
"""
from datetime import datetime, date
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

# 1. Cấu hình giao diện chuẩn Bloomberg Terminal
st.set_page_config(
    page_title="Mini Bloomberg Terminal - Portfolio App",
    page_icon="🖥️",
    layout="wide"
)

st.title("🖥️ Mini Bloomberg Portfolio Terminal")
st.caption("Bước 4 trong lộ trình Portfolio Management App — Trọng số Tương tác & Phân tích Đa chiều")

# 2. Thanh bên (Sidebar) thiết lập tài sản & Trọng số
with st.sidebar:
    st.header("⚙️ Cấu hình Danh mục")
    default_tickers = "AAPL, MSFT, NVDA"
    tickers_input = st.text_input("Mã cổ phiếu danh mục:", default_tickers, help="Nhập các mã phân tách bởi dấu phẩy")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    benchmark_ticker = st.text_input("Benchmark so sánh:", "SPY").strip().upper()

    start_date = st.date_input("Ngày bắt đầu:", date(2023, 1, 1))
    end_date = st.date_input("Ngày kết thúc:", datetime.today().date())
    rf_rate = st.number_input("Lãi suất phi rủi ro (Risk-Free Rate):", value=0.02, step=0.005, format="%.3f")

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

    st.caption("📌 **Trọng số chuẩn hóa (Tổng = 100%):**")
    for k, v in normalized_weights.items():
        st.write(f"- **{k}**: `{v * 100:.1f}%`")

    btn_reload = st.button("🚀 Cập nhật Terminal", use_container_width=True)

# 3. Hàm tải và tính toán dữ liệu
@st.cache_data(ttl=3600)
def fetch_terminal_data(assets: list, benchmark: str, start: str, end: str):
    if not assets:
        return None, "Danh sách tài sản không được để trống."

    symbols_to_download = list(dict.fromkeys(assets + [benchmark]))
    try:
        raw_data = yf.download(symbols_to_download, start=start, end=end, auto_adjust=True)
    except Exception as e:
        return None, f"Lỗi tải dữ liệu: {e}"

    if raw_data.empty:
        return None, "Không tìm thấy dữ liệu cho khoảng thời gian này."

    if "Close" in raw_data:
        close_df = raw_data["Close"].copy()
    else:
        close_df = raw_data.copy()

    if isinstance(close_df, pd.Series):
        close_df = close_df.to_frame(name=symbols_to_download[0])

    if isinstance(close_df.columns, pd.MultiIndex):
        close_df.columns = close_df.columns.get_level_values(-1)

    # Làm sạch dữ liệu
    close_df.dropna(how="all", inplace=True)
    close_df.ffill(inplace=True)
    close_df.dropna(inplace=True)

    if benchmark not in close_df.columns:
        return None, f"Không tải được dữ liệu benchmark {benchmark}."

    for a in assets:
        if a not in close_df.columns:
            return None, f"Không tải được dữ liệu cho mã {a}."

    return close_df, None


def calculate_portfolio_metrics(close_df: pd.DataFrame, assets: list, benchmark: str, weights_dict: dict, rf: float):
    # Lợi nhuận ngày
    daily_returns = close_df.pct_change().dropna()
    weights_vec = np.array([weights_dict[a] for a in assets])

    # Lợi nhuận Portfolio & Benchmark
    port_daily = daily_returns[assets].dot(weights_vec)
    spy_daily = daily_returns[benchmark]

    # Cumulative Returns
    port_cum = (1 + port_daily).cumprod() - 1
    spy_cum = (1 + spy_daily).cumprod() - 1

    # Total Return
    port_total_return = (1 + port_daily).cumprod().iloc[-1] - 1
    spy_total_return = (1 + spy_daily).cumprod().iloc[-1] - 1
    alpha = port_total_return - spy_total_return

    # Volatility & Sharpe
    port_vol = port_daily.std() * np.sqrt(252)
    spy_vol = spy_daily.std() * np.sqrt(252)

    port_sharpe = (port_daily.mean() * 252 - rf) / port_vol if port_vol > 0 else 0.0
    spy_sharpe = (spy_daily.mean() * 252 - rf) / spy_vol if spy_vol > 0 else 0.0

    # Drawdown
    port_wealth = (1 + port_daily).cumprod()
    port_dd = (port_wealth - port_wealth.cummax()) / port_wealth.cummax()
    spy_wealth = (1 + spy_daily).cumprod()
    spy_dd = (spy_wealth - spy_wealth.cummax()) / spy_wealth.cummax()

    # Tín hiệu MA5/MA20 cho từng tài sản
    signals = {}
    for a in assets:
        ma5 = close_df[a].rolling(5).mean()
        ma20 = close_df[a].rolling(20).mean()
        last_ma5 = ma5.iloc[-1]
        last_ma20 = ma20.iloc[-1]
        sig = "BUY" if last_ma5 > last_ma20 else "SELL"
        signals[a] = {
            "Price": close_df[a].iloc[-1],
            "MA5": last_ma5,
            "MA20": last_ma20,
            "Signal": sig
        }

    # Win rate
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
        "signals": signals,
        "win_rate": win_rate,
        "daily_returns": daily_returns,
        "close_df": close_df
    }


# 4. Hiển thị 4 Tab chính
if not tickers:
    st.warning("⚠️ Vui lòng nhập danh sách mã cổ phiếu hợp lệ.")
else:
    with st.spinner("Đang tải dữ liệu terminal và xử lý tín hiệu..."):
        close_data, error_msg = fetch_terminal_data(tickers, benchmark_ticker, str(start_date), str(end_date))

    if error_msg:
        st.error(f"⚠️ {error_msg}")
    else:
        m = calculate_portfolio_metrics(close_data, tickers, benchmark_ticker, normalized_weights, rf_rate)

        # 4 Tab Terminal
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 **Portfolio**",
            "🛡️ **Risk**",
            "📡 **Signals**",
            "🔬 **Backtest**"
        ])

        # ==================== TAB 1: PORTFOLIO ====================
        with tab1:
            st.subheader("📈 Tổng Quan Hiệu Suất Danh Mục vs Benchmark (SPY)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🌟 Portfolio Return", f"{m['port_total_return'] * 100:.2f}%", delta=f"{m['alpha'] * 100:.2f}% vs SPY")
            col2.metric("🏛️ SPY Return", f"{m['spy_total_return'] * 100:.2f}%")
            col3.metric("🔥 Excess Alpha", f"{m['alpha'] * 100:.2f}%", delta=f"{m['alpha'] * 100:.2f}%")
            col4.metric("🎯 Portfolio Sharpe", f"{m['port_sharpe']:.2f}", delta=f"{m['port_sharpe'] - m['spy_sharpe']:.2f}")

            st.divider()

            col_pie, col_chart = st.columns([1, 2])
            with col_pie:
                st.subheader("⚖️ Phân Bổ Trọng Số Tùy Chọn")
                fig_alloc = px.pie(
                    names=list(normalized_weights.keys()),
                    values=[v * 100 for v in normalized_weights.values()],
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig_alloc.update_traces(textposition='inside', textinfo='percent+label')
                fig_alloc.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_alloc, use_container_width=True)

            with col_chart:
                st.subheader(f"📈 Hiệu Suất Lũy Kế: Danh mục vs {benchmark_ticker}")
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=m["port_cum"].index,
                    y=m["port_cum"] * 100,
                    mode="lines",
                    name="🌟 YOUR PORTFOLIO",
                    line=dict(color="#00CC96", width=3)
                ))
                fig_cum.add_trace(go.Scatter(
                    x=m["spy_cum"].index,
                    y=m["spy_cum"] * 100,
                    mode="lines",
                    name=f"🏛️ {benchmark_ticker} (BENCHMARK)",
                    line=dict(color="#636EFA", width=2, dash="dash")
                ))
                fig_cum.update_layout(
                    hovermode="x unified",
                    xaxis_title="Thời gian",
                    yaxis_title="Lợi nhuận tích lũy (%)",
                    height=400,
                    yaxis=dict(ticksuffix="%"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_cum, use_container_width=True)

        # ==================== TAB 2: RISK ====================
        with tab2:
            st.subheader("🛡️ Phân Tích Rủi Ro & Tương Quan Tài Sản")
            col_risk1, col_risk2 = st.columns(2)

            with col_risk1:
                st.subheader("🔥 Ma Trận Tương Quan (Correlation Heatmap)")
                corr_df = m["daily_returns"][tickers + [benchmark_ticker]].corr()
                fig_corr = px.imshow(
                    corr_df,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1.0,
                    zmax=1.0,
                    aspect="auto"
                )
                fig_corr.update_layout(height=380)
                st.plotly_chart(fig_corr, use_container_width=True)

            with col_risk2:
                st.subheader("📉 Biểu Đồ So Sánh Drawdown (%)")
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=m["port_dd"].index,
                    y=m["port_dd"] * 100,
                    mode="lines",
                    name="Portfolio Drawdown",
                    line=dict(color="#EF553B", width=2),
                    fill="tozeroy"
                ))
                fig_dd.add_trace(go.Scatter(
                    x=m["spy_dd"].index,
                    y=m["spy_dd"] * 100,
                    mode="lines",
                    name=f"{benchmark_ticker} Drawdown",
                    line=dict(color="#AB63FA", width=1.5, dash="dot")
                ))
                fig_dd.update_layout(
                    xaxis_title="Thời gian",
                    yaxis_title="Sụt giảm (%)",
                    height=380,
                    yaxis=dict(ticksuffix="%"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_dd, use_container_width=True)

            st.divider()
            st.subheader("📋 Bảng So Sánh Chỉ Số Biến Động")
            risk_summary = pd.DataFrame({
                "Chỉ số": ["Độ Biến Động (Volatility Năm)", "Mức Sụt Giảm Lớn Nhất (Max Drawdown)", "Sharpe Ratio"],
                "Portfolio": [f"{m['port_vol'] * 100:.2f}%", f"{m['port_dd'].min() * 100:.2f}%", f"{m['port_sharpe']:.2f}"],
                f"Benchmark ({benchmark_ticker})": [f"{m['spy_vol'] * 100:.2f}%", f"{m['spy_dd'].min() * 100:.2f}%", f"{m['spy_sharpe']:.2f}"]
            })
            st.dataframe(risk_summary.set_index("Chỉ số"), use_container_width=True)

        # ==================== TAB 3: SIGNALS ====================
        with tab3:
            st.subheader("📡 Tín Hiệu Giao Dịch MA5 / MA20 Từng Cổ Phiếu")
            sig_cols = st.columns(len(tickers))
            for i, t in enumerate(tickers):
                info = m["signals"][t]
                with sig_cols[i]:
                    badge = "🟢 BUY" if info["Signal"] == "BUY" else "🔴 SELL"
                    st.metric(f"Mã: {t}", f"${info['Price']:.2f}", delta=badge)
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
            st.subheader(f"🔬 Hiệu Suất Lịch Sử Chi Tiết vs {benchmark_ticker}")
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            col_b1.metric("🏆 Win Rate (% Ngày Dương)", f"{m['win_rate']:.1f}%")
            col_b2.metric("📅 Tổng Số Phiên Giao Dịch", f"{len(m['port_daily'])}")
            col_b3.metric("🔼 Lợi Nhuận Ngày Tốt Nhất", f"{m['port_daily'].max() * 100:.2f}%")
            col_b4.metric("🔽 Lợi Nhuận Ngày Kém Nhất", f"{m['port_daily'].min() * 100:.2f}%")

            st.divider()
            st.subheader("📊 Histogram So Sánh Phân Phối Lợi Nhuận Ngày (Portfolio vs SPY)")
            compare_df = pd.DataFrame({
                "Portfolio": m["port_daily"] * 100,
                benchmark_ticker: m["spy_daily"] * 100
            })
            fig_hist = px.histogram(
                compare_df,
                barmode="overlay",
                nbins=60,
                opacity=0.6,
                labels={"value": "Lợi nhuận ngày (%)", "variable": "Tài sản"},
                color_discrete_map={"Portfolio": "#00CC96", benchmark_ticker: "#636EFA"}
            )
            fig_hist.update_layout(height=400, xaxis_title="Lợi nhuận ngày (%)", yaxis_title="Tần suất")
            st.plotly_chart(fig_hist, use_container_width=True)
