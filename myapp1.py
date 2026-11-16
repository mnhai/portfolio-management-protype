"""Bước 2 — Personal Trading Analytics App (myapp1.py)
Tải dữ liệu cổ phiếu, tính chỉ báo kỹ thuật (MA20, MA50), các chỉ số hiệu suất
(Total Return, Volatility, Sharpe Ratio), vẽ biểu đồ giá, drawdown, phân phối lợi nhuận,
và tín hiệu giao dịch BUY/SELL theo MA5/MA20 crossover.
"""
from datetime import datetime, date
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

# 1. Cấu hình trang
st.set_page_config(
    page_title="Personal Trading Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Personal Trading Analytics Dashboard")
st.caption("Bước 2 trong lộ trình Portfolio Management App — Phân tích kỹ thuật & Hiệu suất cá nhân")

# 2. Thanh bên (Sidebar) thiết lập
with st.sidebar:
    st.header("⚙️ Thiết lập Phân tích")
    ticker = st.text_input("Mã cổ phiếu (Ticker):", "AAPL").strip().upper()
    start_date = st.date_input("Ngày bắt đầu:", date(2023, 1, 1))
    end_date = st.date_input("Ngày kết thúc:", datetime.today().date())
    risk_free_rate = st.number_input("Lãi suất phi rủi ro (Risk-free rate):", value=0.02, step=0.005, format="%.3f")
    btn_run = st.button("🚀 Cập nhật Phân tích", use_container_width=True)

# 3. Hàm tải và xử lý dữ liệu
@st.cache_data(ttl=3600)
def fetch_and_calculate(symbol: str, start: str, end: str, rf: float = 0.02):
    try:
        data = yf.download(symbol, start=start, end=end, auto_adjust=True, multi_level_index=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
    except Exception as e:
        return None, f"Lỗi kết nối yfinance: {e}"

    if data.empty:
        return None, "Không tìm thấy dữ liệu cho mã cổ phiếu này."

    df = data.copy()
    
    # Chỉ báo kỹ thuật
    df["Daily_Return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()

    # Drawdown
    cum_max = df["Close"].cummax()
    df["Drawdown"] = (df["Close"] - cum_max) / cum_max

    # Tín hiệu giao dịch (MA5 / MA20 crossover)
    df["Signal"] = np.where(df["MA5"] > df["MA20"], "BUY", "SELL")
    
    # Tính các chỉ số hiệu suất
    clean_returns = df["Daily_Return"].dropna()
    total_return = (df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1
    annualized_volatility = clean_returns.std() * np.sqrt(252)
    
    # Annualized Sharpe Ratio
    if annualized_volatility > 0:
        annualized_return = clean_returns.mean() * 252
        sharpe_ratio = (annualized_return - rf) / annualized_volatility
    else:
        sharpe_ratio = 0.0

    max_drawdown = df["Drawdown"].min()
    current_signal = df["Signal"].iloc[-1]

    metrics = {
        "last_price": float(df["Close"].iloc[-1]),
        "total_return": total_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "current_signal": current_signal,
    }

    return df, metrics


# 4. Hiển thị nội dung
if ticker:
    with st.spinner(f"Đang tải dữ liệu và phân tích cho {ticker}..."):
        df, metrics_or_err = fetch_and_calculate(ticker, str(start_date), str(end_date), risk_free_rate)

    if df is None:
        st.error(f"⚠️ {metrics_or_err}")
    else:
        m = metrics_or_err
        
        # 4 Hộp chỉ số (KPI Cards)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Giá Đóng Cửa", f"${m['last_price']:.2f}")
        col1.metric("📈 Total Return", f"{m['total_return'] * 100:.2f}%", delta=f"{m['total_return'] * 100:.2f}%")
        
        col2.metric("📉 Volatility (Năm)", f"{m['annualized_volatility'] * 100:.2f}%")
        col2.metric("⚡ Max Drawdown", f"{m['max_drawdown'] * 100:.2f}%")

        col3.metric("🎯 Sharpe Ratio", f"{m['sharpe_ratio']:.2f}")
        
        signal_badge = "🟢 BUY" if m["current_signal"] == "BUY" else "🔴 SELL"
        col4.metric("📡 Tín hiệu MA5/MA20", signal_badge)

        st.divider()

        # Biểu đồ giá + Đường trung bình động
        st.subheader("📈 Biểu đồ Giá & Đường Trung Bình Động (MA20, MA50, MA5)")
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close Price", line=dict(color="#1f77b4", width=2)))
        fig_price.add_trace(go.Scatter(x=df.index, y=df["MA5"], mode="lines", name="MA 5 (Ngắn hạn)", line=dict(color="#ff7f0e", width=1.5, dash="dot")))
        fig_price.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA 20 (Trung hạn)", line=dict(color="#2ca02c", width=1.8)))
        fig_price.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA 50 (Dài hạn)", line=dict(color="#d62728", width=1.8)))
        fig_price.update_layout(
            hovermode="x unified",
            xaxis_title="Thời gian",
            yaxis_title="Giá (USD)",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_price, use_container_width=True)

        # 2 Cột hiển thị Drawdown và Phân phối lợi nhuận
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("📉 Biểu đồ Drawdown (%)")
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=df.index,
                y=df["Drawdown"] * 100,
                fill="tozeroy",
                mode="lines",
                name="Drawdown",
                line=dict(color="red", width=1.5)
            ))
            fig_dd.update_layout(
                xaxis_title="Thời gian",
                yaxis_title="Mức sụt giảm (%)",
                height=350,
                yaxis=dict(ticksuffix="%")
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        with col_right:
            st.subheader("📊 Phân Phối Lợi Nhuận Hàng Ngày")
            fig_hist = px.histogram(
                df["Daily_Return"].dropna() * 100,
                nbins=50,
                labels={"value": "Lợi nhuận ngày (%)"},
                title=f"Phân phối Daily Return ({ticker})",
                color_discrete_sequence=["#17becf"]
            )
            fig_hist.update_layout(height=350, showlegend=False, yaxis_title="Tần suất")
            st.plotly_chart(fig_hist, use_container_width=True)

        # Bảng dữ liệu gần nhất
        with st.expander("📋 Xem dữ liệu bảng chi tiết gần nhất"):
            st.dataframe(df[["Close", "MA5", "MA20", "MA50", "Daily_Return", "Drawdown", "Signal"]].tail(20), use_container_width=True)
