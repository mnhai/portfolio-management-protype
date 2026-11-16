"""Bước 8 — Bull/Bear Regime Switching Strategy (myapp10.py)
Xây dựng cỗ máy nhận diện chế độ thị trường (Market Regime Detection Engine) trên SPY:
- Trend: MA50 so với MA200 (1 nếu MA50 > MA200, ngược lại 0).
- Momentum: Lợi nhuận 12 ngày (1 nếu > 0, ngược lại 0).
- Volatility: Biến động 20 ngày so với trung vị 60 ngày (1 nếu High Vol = Risk-off, ngược lại 0).
- Regime Score = Trend + Momentum - Volatility.
- Gán nhãn chế độ thị trường:
  * Score >= 1: BULL (Tăng trưởng - Màu xanh)
  * Score == 0: NEUTRAL (Trung lập - Màu vàng)
  * Score <= -1: BEAR (Phòng thủ - Màu đỏ)
- Chiến lược tự động luân chuyển phân bổ tài sản (Regime Switching Strategy):
  * BULL: 100% Growth Portfolio (User/Optimized Assets)
  * NEUTRAL: 50% Growth Portfolio + 50% Defensive (SPY/Cash)
  * BEAR: 100% Defensive (Cash / SPY)
- Trực quan hóa bản đồ chế độ thị trường: SPY Price Regime Map (Xanh/Vàng/Đỏ).
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
st.caption("Bước 8 trong lộ trình Portfolio Management App — Chiến lược Luân chuyển Phân bổ Động theo Chế độ Thị trường")

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

    st.subheader("⚖️ Trọng Số Danh Mục Tăng Trưởng (Growth Weights)")
    raw_weights = {}
    default_w = 1.0 / len(tickers) if tickers else 1.0
    for t in tickers:
        raw_weights[t] = st.slider(f"Trọng số {t}:", 0.0, 1.0, float(default_w), 0.05)

    total_raw = sum(raw_weights.values())
    normalized_weights = {k: v / total_raw for k, v in raw_weights.items()} if total_raw > 0 else {k: 1.0 / len(tickers) for k in tickers}

    btn_reload = st.button("🚀 Cập nhật Chiến Lược Regime", use_container_width=True)


# 3. Hàm tải dữ liệu và tính toán Regime
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


def compute_market_regimes(spy_close: pd.Series):
    """
    Tính toán tín hiệu chế độ thị trường dựa trên SPY:
    1. Trend: MA50 > MA200
    2. Momentum: Return 12 ngày > 0
    3. Volatility: Vol 20 ngày > median 60 ngày (Risk-off)
    """
    spy_daily = spy_close.pct_change()
    ma50 = spy_close.rolling(50).mean()
    ma200 = spy_close.rolling(200).mean()

    # 1. Trend Signal (1 if Bull trend, 0 otherwise)
    trend_signal = np.where(ma50 > ma200, 1, 0)

    # 2. Momentum Signal (1 if > 0, 0 otherwise)
    mom12 = (spy_close / spy_close.shift(12)) - 1
    mom_signal = np.where(mom12 > 0, 1, 0)

    # 3. Volatility Signal (1 if Risk-Off / High Vol, 0 otherwise)
    vol20 = spy_daily.rolling(20).std() * np.sqrt(252)
    vol_median60 = vol20.rolling(60).median()
    vol_risk_off = np.where(vol20 > vol_median60, 1, 0)

    # Regime Score = Trend + Momentum - Volatility
    regime_score = trend_signal + mom_signal - vol_risk_off

    # Gán nhãn: BULL (>= 1), NEUTRAL (0), BEAR (<= -1)
    conditions = [
        regime_score >= 1,
        regime_score == 0,
        regime_score <= -1
    ]
    choices = ["BULL", "NEUTRAL", "BEAR"]
    regime_labels = pd.Series(np.select(conditions, choices, default="NEUTRAL"), index=spy_close.index)

    regime_df = pd.DataFrame({
        "Close": spy_close,
        "MA50": ma50,
        "MA200": ma200,
        "Trend_Signal": trend_signal,
        "Mom12": mom12,
        "Mom_Signal": mom_signal,
        "Vol20": vol20,
        "Vol_Median60": vol_median60,
        "Vol_Risk_Off": vol_risk_off,
        "Regime_Score": regime_score,
        "Regime": regime_labels
    })

    return regime_df.dropna()


def simulate_regime_switching_strategy(close_df: pd.DataFrame, assets: list, benchmark: str, weights_dict: dict, regime_df: pd.DataFrame):
    """
    Mô phỏng chiến lược tự động luân chuyển:
    - Khi BULL: 100% Growth Portfolio (User Assets)
    - Khi NEUTRAL: 50% Growth + 50% Defensive (SPY)
    - Khi BEAR: 100% Defensive (SPY)
    """
    daily_returns = close_df.pct_change().dropna()
    common_idx = daily_returns.index.intersection(regime_df.index)

    daily_returns = daily_returns.loc[common_idx]
    regime_df = regime_df.loc[common_idx]

    weights_vec = np.array([weights_dict[a] for a in assets])
    growth_daily = daily_returns[assets].dot(weights_vec)
    defensive_daily = daily_returns[benchmark]

    # Quyết định phân bổ theo chế độ ngày hôm trước (tránh look-ahead bias)
    lagged_regime = regime_df["Regime"].shift(1).fillna("NEUTRAL")

    strategy_daily = pd.Series(index=common_idx, dtype=float)
    for t in common_idx:
        reg = lagged_regime.loc[t]
        if reg == "BULL":
            strategy_daily.loc[t] = growth_daily.loc[t]
        elif reg == "NEUTRAL":
            strategy_daily.loc[t] = 0.5 * growth_daily.loc[t] + 0.5 * defensive_daily.loc[t]
        else:  # BEAR
            strategy_daily.loc[t] = defensive_daily.loc[t]

    # Tính tích lũy
    strategy_cum = (1 + strategy_daily).cumprod() - 1
    growth_cum = (1 + growth_daily).cumprod() - 1
    spy_cum = (1 + defensive_daily).cumprod() - 1

    return {
        "strategy_daily": strategy_daily,
        "strategy_cum": strategy_cum,
        "growth_daily": growth_daily,
        "growth_cum": growth_cum,
        "spy_daily": defensive_daily,
        "spy_cum": spy_cum,
        "regime_df": regime_df,
        "lagged_regime": lagged_regime
    }


# 4. Hiển thị Giao diện
if not tickers:
    st.warning("⚠️ Vui lòng nhập danh sách mã hợp lệ.")
else:
    with st.spinner("Đang tính toán chế độ thị trường và chạy chiến lược Regime Switching..."):
        close_data, err = fetch_terminal_data(tickers, benchmark_ticker, str(start_date), str(end_date))

    if err:
        st.error(f"⚠️ {err}")
    else:
        regime_data = compute_market_regimes(close_data[benchmark_ticker])
        sim = simulate_regime_switching_strategy(close_data, tickers, benchmark_ticker, normalized_weights, regime_data)

        tab1, tab2, tab3 = st.tabs([
            "🧭 **Regime Switching Strategy**",
            "🗺️ **SPY Price Regime Map**",
            "📊 **Regime Signals Breakdown**"
        ])

        # TAB 1: STRATEGY OVERVIEW
        with tab1:
            st.subheader("🧭 Chiến Lược Luân Chuyển Tự Động Theo Chế Độ Thị Trường")
            latest_row = regime_data.iloc[-1]
            current_regime = latest_row["Regime"]

            # Badge trạng thái hiện tại
            regime_colors = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}
            icon = regime_colors.get(current_regime, "⚪")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🌐 Chế Độ Thị Trường", f"{icon} {current_regime}")
            c2.metric("🔢 Regime Score", f"{int(latest_row['Regime_Score'])}")
            
            strat_total = sim["strategy_cum"].iloc[-1]
            growth_total = sim["growth_cum"].iloc[-1]
            c3.metric("🚀 Strategy Return", f"{strat_total * 100:.2f}%", delta=f"{(strat_total - growth_total)*100:.2f}% vs Buy&Hold")
            c4.metric("🏛️ SPY Return", f"{sim['spy_cum'].iloc[-1] * 100:.2f}%")

            st.divider()

            # Nhận định phân bổ hiện tại
            if current_regime == "BULL":
                alloc_msg = "Thị trường trong trạng thái **BULL**: Phân bổ **100% vào Danh mục Tăng trưởng (Growth Assets)** để tối ưu hóa lợi nhuận."
            elif current_regime == "NEUTRAL":
                alloc_msg = "Thị trường trong trạng thái **NEUTRAL**: Phân bổ **50% Tăng trưởng + 50% Phòng thủ (SPY)** để cân bằng rủi ro."
            else:
                alloc_msg = "Thị trường trong trạng thái **BEAR (Risk-Off)**: Phân bổ **100% vào Phòng thủ (SPY/Tiền mặt)** nhằm bảo toàn vốn."

            st.info(f"💡 **Khuyến Nghị Phân Bổ Hiện Tại:** {alloc_msg}")

            st.divider()

            # Biểu đồ đường tích lũy so sánh
            st.subheader("📈 Hiệu Suất So Sánh: Regime Switching Strategy vs Static Buy & Hold vs SPY")
            fig_strat = go.Figure()
            fig_strat.add_trace(go.Scatter(
                x=sim["strategy_cum"].index,
                y=sim["strategy_cum"] * 100,
                mode="lines",
                name="🛡️ REGIME SWITCHING STRATEGY",
                line=dict(color="#00CC96", width=3)
            ))
            fig_strat.add_trace(go.Scatter(
                x=sim["growth_cum"].index,
                y=sim["growth_cum"] * 100,
                mode="lines",
                name="👤 STATIC GROWTH PORTFOLIO",
                line=dict(color="#636EFA", width=2)
            ))
            fig_strat.add_trace(go.Scatter(
                x=sim["spy_cum"].index,
                y=sim["spy_cum"] * 100,
                mode="lines",
                name=f"🏛️ {benchmark_ticker} BUY & HOLD",
                line=dict(color="#EF553B", width=1.5, dash="dash")
            ))
            fig_strat.update_layout(
                height=450,
                xaxis_title="Thời gian",
                yaxis_title="Lợi nhuận tích lũy (%)",
                yaxis=dict(ticksuffix="%"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            st.plotly_chart(fig_strat, use_container_width=True)

        # TAB 2: SPY REGIME MAP (XANH / VÀNG / ĐỎ)
        with tab2:
            st.subheader("🗺️ Bản Đồ Chế Độ Thị Trường (SPY Price Regime Map)")
            st.caption("Biểu đồ giá SPY được phân đoạn màu sắc theo từng chế độ: Xanh (BULL), Vàng (NEUTRAL), Đỏ (BEAR)")

            fig_map = go.Figure()
            color_map = {"BULL": "#00CC96", "NEUTRAL": "#FFA15A", "BEAR": "#EF553B"}

            # Vẽ giá SPY nền
            fig_map.add_trace(go.Scatter(
                x=regime_data.index,
                y=regime_data["Close"],
                mode="lines",
                name="SPY Price",
                line=dict(color="rgba(150, 150, 150, 0.4)", width=1.5)
            ))

            # Vẽ điểm phân loại theo màu sắc từng chế độ
            for reg in ["BULL", "NEUTRAL", "BEAR"]:
                sub = regime_data[regime_data["Regime"] == reg]
                fig_map.add_trace(go.Scatter(
                    x=sub.index,
                    y=sub["Close"],
                    mode="markers",
                    name=f"Chế độ {reg}",
                    marker=dict(color=color_map[reg], size=5)
                ))

            # Vẽ thêm MA50 và MA200
            fig_map.add_trace(go.Scatter(x=regime_data.index, y=regime_data["MA50"], name="MA50", line=dict(color="#1f77b4", width=1.5, dash="dot")))
            fig_map.add_trace(go.Scatter(x=regime_data.index, y=regime_data["MA200"], name="MA200", line=dict(color="#9467bd", width=1.5)))

            fig_map.update_layout(height=480, xaxis_title="Thời gian", yaxis_title="SPY Price (USD)", hovermode="x unified")
            st.plotly_chart(fig_map, use_container_width=True)

        # TAB 3: SIGNALS BREAKDOWN
        with tab3:
            st.subheader("📊 Chi Tiết 3 Tín Hiệu Thành Phần Tạo Nên Regime Score")
            t_col, m_col, v_col = st.columns(3)
            with t_col:
                st.write("📈 **1. Tín hiệu Xu Hướng (Trend)**")
                st.write(f"- MA50: **${latest_row['MA50']:.2f}**")
                st.write(f"- MA200: **${latest_row['MA200']:.2f}**")
                st.metric("Điểm Trend (0/1)", f"{int(latest_row['Trend_Signal'])}")

            with m_col:
                st.write("⚡ **2. Tín hiệu Đà (Momentum 12d)**")
                st.write(f"- Lợi nhuận 12d: **{latest_row['Mom12']*100:.2f}%**")
                st.metric("Điểm Momentum (0/1)", f"{int(latest_row['Mom_Signal'])}")

            with v_col:
                st.write("📉 **3. Tín hiệu Biến Động (Volatility)**")
                st.write(f"- Vol 20d: **{latest_row['Vol20']*100:.2f}%**")
                st.write(f"- Median 60d: **{latest_row['Vol_Median60']*100:.2f}%**")
                st.metric("Risk-off Phạt (-1/0)", f"-{int(latest_row['Vol_Risk_Off'])}")

            st.divider()
            st.subheader("📋 Bảng Lịch Sử Regime Gần Nhất")
            st.dataframe(regime_data[["Close", "Regime_Score", "Regime", "Trend_Signal", "Mom_Signal", "Vol_Risk_Off"]].tail(20), use_container_width=True)
