"""Bước 3 — Multi-Asset Equally Weighted Portfolio App (myapp2.py)
Cho phép nhập nhiều mã cổ phiếu (vd: AAPL, MSFT, NVDA), tự động chia đều trọng số (Equal Weight).
Tính toán hiệu suất danh mục, biểu đồ phân bổ (Donut), hiệu suất tích lũy so với từng tài sản,
biểu đồ drawdown và phân phối lợi nhuận danh mục.
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
    page_title="Multi-Asset Equally Weighted Portfolio",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Multi-Asset Equally Weighted Portfolio Dashboard")
st.caption("Bước 3 trong lộ trình Portfolio Management App — Danh mục Đa Tài Sản với Trọng Số Đồng Đều (1/N)")

# 2. Thanh bên (Sidebar) thiết lập
with st.sidebar:
    st.header("⚙️ Cấu hình Danh mục")
    default_tickers = "AAPL, MSFT, NVDA"
    tickers_input = st.text_input(
        "Nhập danh sách mã cổ phiếu (cách nhau bởi dấu phẩy):",
        default_tickers,
        help="Ví dụ: AAPL, MSFT, NVDA, GOOGL"
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    start_date = st.date_input("Ngày bắt đầu:", date(2023, 1, 1))
    end_date = st.date_input("Ngày kết thúc:", datetime.today().date())
    risk_free_rate = st.number_input("Lãi suất phi rủi ro (Risk-free rate):", value=0.02, step=0.005, format="%.3f")

    st.info(f"📊 Số lượng tài sản: **{len(tickers)}**\n\n⚖️ Trọng số mỗi tài sản: **{100/len(tickers):.2f}%**" if tickers else "Vui lòng nhập ít nhất 1 mã.")
    btn_update = st.button("🚀 Cập nhật Danh mục", use_container_width=True)

# 3. Hàm tải và tính toán hiệu suất danh mục
@st.cache_data(ttl=3600)
def fetch_portfolio_data(symbols: list, start: str, end: str, rf: float = 0.02):
    if not symbols:
        return None, "Danh sách mã cổ phiếu trống."

    try:
        raw_data = yf.download(symbols, start=start, end=end, auto_adjust=True)
    except Exception as e:
        return None, f"Lỗi tải dữ liệu: {e}"

    if raw_data.empty:
        return None, "Không tìm thấy dữ liệu cho danh sách mã này."

    # Trích xuất Close price
    if "Close" in raw_data:
        close_df = raw_data["Close"].copy()
    else:
        close_df = raw_data.copy()

    # Đảm bảo dạng DataFrame khi chỉ có 1 mã
    if isinstance(close_df, pd.Series):
        close_df = close_df.to_frame(name=symbols[0])

    # Xử lý MultiIndex columns nếu có
    if isinstance(close_df.columns, pd.MultiIndex):
        close_df.columns = close_df.columns.get_level_values(-1)

    # Loại bỏ missing values
    close_df.dropna(how="all", inplace=True)
    close_df.ffill(inplace=True)
    close_df.dropna(inplace=True)

    if close_df.empty or len(close_df) < 5:
        return None, "Dữ liệu quá ngắn hoặc không đủ ngày giao dịch để tính toán."

    # Lợi nhuận hàng ngày từng tài sản
    daily_returns = close_df.pct_change().dropna()
    num_assets = len(close_df.columns)
    weights = np.ones(num_assets) / num_assets

    # Lợi nhuận hàng ngày của danh mục (Equally Weighted: w_i = 1/N)
    portfolio_daily_returns = daily_returns.dot(weights)
    portfolio_daily_returns.name = "Portfolio"

    # Chỉ số tích lũy (Wealth Index)
    portfolio_cum_returns = (1 + portfolio_daily_returns).cumprod() - 1
    assets_cum_returns = (1 + daily_returns).cumprod() - 1

    # Drawdown danh mục
    wealth_index = (1 + portfolio_daily_returns).cumprod()
    cum_max = wealth_index.cummax()
    portfolio_drawdown = (wealth_index - cum_max) / cum_max

    # Metrics tổng hợp
    total_return = wealth_index.iloc[-1] - 1
    annualized_volatility = portfolio_daily_returns.std() * np.sqrt(252)
    annualized_return = portfolio_daily_returns.mean() * 252

    if annualized_volatility > 0:
        sharpe_ratio = (annualized_return - rf) / annualized_volatility
    else:
        sharpe_ratio = 0.0

    max_drawdown = portfolio_drawdown.min()

    # Tính lợi nhuận và biến động từng tài sản để so sánh
    asset_metrics = {}
    for col in daily_returns.columns:
        asset_cum = (1 + daily_returns[col]).cumprod().iloc[-1] - 1
        asset_vol = daily_returns[col].std() * np.sqrt(252)
        asset_metrics[col] = {
            "Total_Return": asset_cum,
            "Volatility": asset_vol,
            "Weight": weights[0]
        }

    results = {
        "close_df": close_df,
        "daily_returns": daily_returns,
        "portfolio_daily_returns": portfolio_daily_returns,
        "portfolio_cum_returns": portfolio_cum_returns,
        "assets_cum_returns": assets_cum_returns,
        "portfolio_drawdown": portfolio_drawdown,
        "total_return": total_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "tickers": list(close_df.columns),
        "weights": weights,
        "asset_metrics": asset_metrics,
    }

    return results, None


# 4. Hiển thị Dashboard
if not tickers:
    st.warning("⚠️ Vui lòng nhập ít nhất một mã cổ phiếu hợp lệ ở thanh bên trái.")
else:
    with st.spinner(f"Đang tính toán danh mục cho: {', '.join(tickers)}..."):
        data, err = fetch_portfolio_data(tickers, str(start_date), str(end_date), risk_free_rate)

    if err:
        st.error(f"⚠️ {err}")
    else:
        # KPI Cards tổng quan danh mục
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📈 Portfolio Return", f"{data['total_return'] * 100:.2f}%", delta=f"{data['total_return'] * 100:.2f}%")
        col2.metric("📉 Annualized Volatility", f"{data['annualized_volatility'] * 100:.2f}%")
        col3.metric("🎯 Sharpe Ratio (Năm)", f"{data['sharpe_ratio']:.2f}")
        col4.metric("⚡ Max Drawdown", f"{data['max_drawdown'] * 100:.2f}%")

        st.divider()

        # Phần 1: Biểu đồ phân bổ & Biểu đồ hiệu suất so sánh
        col_pie, col_perf = st.columns([1, 2])

        with col_pie:
            st.subheader("⚖️ Phân Bổ Trọng Số (Donut)")
            fig_alloc = px.pie(
                names=data["tickers"],
                values=data["weights"] * 100,
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_alloc.update_traces(textposition='inside', textinfo='percent+label')
            fig_alloc.update_layout(height=420, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2))
            st.plotly_chart(fig_alloc, use_container_width=True)

        with col_perf:
            st.subheader("📈 Hiệu Suất Tích Lũy (Portfolio vs Cổ Phiếu Thành Phần)")
            fig_perf = go.Figure()
            # Đường Danh mục
            fig_perf.add_trace(go.Scatter(
                x=data["portfolio_cum_returns"].index,
                y=data["portfolio_cum_returns"] * 100,
                mode="lines",
                name="🌟 EQUAL-WEIGHT PORTFOLIO",
                line=dict(color="#00CC96", width=3)
            ))
            # Đường các cổ phiếu con
            for ticker_name in data["tickers"]:
                fig_perf.add_trace(go.Scatter(
                    x=data["assets_cum_returns"].index,
                    y=data["assets_cum_returns"][ticker_name] * 100,
                    mode="lines",
                    name=ticker_name,
                    line=dict(width=1.5, dash="dot")
                ))
            fig_perf.update_layout(
                hovermode="x unified",
                xaxis_title="Thời gian",
                yaxis_title="Lợi nhuận tích lũy (%)",
                height=420,
                yaxis=dict(ticksuffix="%"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_perf, use_container_width=True)

        st.divider()

        # Phần 2: Drawdown & Histogram phân phối lợi nhuận
        col_dd, col_hist = st.columns(2)

        with col_dd:
            st.subheader("📉 Biểu Đồ Drawdown Danh Mục")
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=data["portfolio_drawdown"].index,
                y=data["portfolio_drawdown"] * 100,
                fill="tozeroy",
                mode="lines",
                name="Portfolio Drawdown",
                line=dict(color="#EF553B", width=1.5)
            ))
            fig_dd.update_layout(
                xaxis_title="Thời gian",
                yaxis_title="Sụt giảm từ đỉnh (%)",
                height=350,
                yaxis=dict(ticksuffix="%")
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        with col_hist:
            st.subheader("📊 Phân Phối Lợi Nhuận Ngày (Daily Returns)")
            fig_hist = px.histogram(
                data["portfolio_daily_returns"] * 100,
                nbins=50,
                labels={"value": "Lợi nhuận ngày (%)"},
                color_discrete_sequence=["#636EFA"]
            )
            fig_hist.update_layout(height=350, showlegend=False, yaxis_title="Số phiên giao dịch")
            st.plotly_chart(fig_hist, use_container_width=True)

        # Bảng so sánh chi tiết từng tài sản
        with st.expander("📋 Bảng so sánh chi tiết các tài sản trong danh mục"):
            summary_table = pd.DataFrame.from_dict(data["asset_metrics"], orient="index")
            summary_table["Weight"] = summary_table["Weight"].map(lambda x: f"{x * 100:.2f}%")
            summary_table["Total_Return"] = summary_table["Total_Return"].map(lambda x: f"{x * 100:.2f}%")
            summary_table["Volatility"] = summary_table["Volatility"].map(lambda x: f"{x * 100:.2f}%")
            st.dataframe(summary_table, use_container_width=True)
