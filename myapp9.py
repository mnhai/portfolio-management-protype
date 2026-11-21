import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(
    page_title="Bloomberg Portfolio Terminal",
    layout="wide"
)

st.title("📈 Bloomberg-Style Portfolio Terminal")

# ====================
# INPUTS
# ====================
st.sidebar.header("Portfolio Builder")

tickers = st.sidebar.text_input(
    "Tickers",
    "AAPL,MSFT,NVDA"
)

ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

start_date = st.sidebar.date_input(
    "Start Date",
    pd.Timestamp("2023-01-01")
)

# ====================
# WEIGHTS
# ====================
st.sidebar.subheader("Weights")

weights = []
for t in ticker_list:
    w = st.sidebar.slider(t, 0.0, 1.0, 1 / len(ticker_list))
    weights.append(w)

weights = np.array(weights)

if weights.sum() == 0:
    weights = np.ones(len(weights)) / len(weights)
else:
    weights = weights / weights.sum()

allocation = pd.DataFrame({
    "Ticker": ticker_list,
    "Weight": weights
})

# ====================
# DATA
# ====================
data = yf.download(
    ticker_list,
    start=start_date,
    auto_adjust=True
)["Close"]

if isinstance(data, pd.Series):
    data = data.to_frame()

data = data.dropna(how="all")

# ====================
# PORTFOLIO ENGINE
# ====================
returns = data.pct_change().dropna()
portfolio_returns = returns.dot(weights)

cumulative = (1 + portfolio_returns).cumprod()

# ====================
# PORTFOLIO OPTIMIZATION (MAX SHARPE)
# ====================
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

num_assets = len(ticker_list)


def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = returns / volatility
    return returns, volatility, sharpe


def neg_sharpe(weights):
    return -portfolio_performance(weights, mean_returns, cov_matrix)[2]


# constraints
constraints = (
    {"type": "eq", "fun": lambda x: np.sum(x) - 1}
)

# bounds (long-only + optional cap)
max_weight = 0.5
bounds = tuple((0, max_weight) for _ in range(num_assets))

# initial guess
init_guess = np.array([1 / num_assets] * num_assets)

opt_result = minimize(
    neg_sharpe,
    init_guess,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

opt_weights = opt_result.x

opt_return, opt_vol, opt_sharpe = portfolio_performance(
    opt_weights, mean_returns, cov_matrix
)

optimized_returns = returns.dot(opt_weights)
optimized_cum = (1 + optimized_returns).cumprod()

# ====================
# SPY BENCHMARK
# ====================
spy = yf.download("SPY", start=start_date, auto_adjust=True)["Close"].squeeze()
spy_returns = spy.pct_change().dropna()

# align to portfolio
aligned = pd.concat([portfolio_returns, spy_returns], axis=1, join="inner")
aligned.columns = ["portfolio", "spy"]

portfolio_aligned = aligned["portfolio"]
spy_aligned = aligned["spy"]

spy_cum = (1 + spy_aligned).cumprod()
cumulative = (1 + portfolio_aligned).cumprod()

# risk metrics for SPY
spy_vol = spy_returns.rolling(20).std() * np.sqrt(252)
spy_drawdown = spy_cum / spy_cum.cummax() - 1

# ====================
# METRICS
# ====================
portfolio_total_return = float((cumulative.iloc[-1] - 1) * 100)
spy_total_return = float((spy_cum.iloc[-1] - 1) * 100)

alpha = float(portfolio_total_return - spy_total_return)

# ====================
# RISK ENGINE (PORTFOLIO)
# ====================
corr = returns.corr()
rolling_vol = portfolio_returns.rolling(20).std() * np.sqrt(252)
drawdown = cumulative / cumulative.cummax() - 1

# ====================
# ROLLING BETA ENGINE
# ====================
rolling_window = 20

cov = portfolio_aligned.rolling(rolling_window).cov(spy_aligned)
var = spy_aligned.rolling(rolling_window).var()

rolling_beta = cov / var
rolling_beta = rolling_beta.dropna()

# rolling alpha (excess returns)
rolling_alpha = (
    portfolio_aligned.rolling(rolling_window).mean()
    - rolling_beta * spy_aligned.rolling(rolling_window).mean()
)

# ====================
# FACTOR ENGINE (MARKET + MOMENTUM)
# ====================

# --------
# MOMENTUM FACTOR (SPY PROXY)
# --------
momentum_window = 12

spy_momentum = spy_aligned.pct_change(momentum_window)

# align everything
factor_df = pd.concat([
    portfolio_aligned,
    spy_aligned,
    spy_momentum
], axis=1).dropna()

factor_df.columns = ["portfolio", "market", "momentum"]

# --------
# ROLLING MOMENTUM EXPOSURE
# --------
rolling_mom_corr = factor_df["portfolio"].rolling(20).corr(factor_df["momentum"])

# --------
# ROLLING MARKET BETA (already exists but recomputed cleanly here for consistency)
# --------
rolling_mkt_beta = (
    factor_df["portfolio"].rolling(20).cov(factor_df["market"])
    / factor_df["market"].rolling(20).var()
)

rolling_mkt_beta = rolling_mkt_beta.dropna()
rolling_mom_corr = rolling_mom_corr.dropna()

# ====================
# STRATEGY ENGINE
# ====================
signal_list = []

for t in ticker_list:
    px_data = yf.download(t, start=start_date, auto_adjust=True)["Close"]

    ma20 = px_data.rolling(20).mean()
    ma50 = px_data.rolling(50).mean()

    signal_list.append((ma20 > ma50).astype(int))

signal_df = pd.concat(signal_list, axis=1)
signal_df.columns = ticker_list
signal_df = signal_df.dropna()

portfolio_signal_series = signal_df.mean(axis=1)
final_signal = "BUY" if portfolio_signal_series.iloc[-1] > 0.5 else "SELL"

# ====================
# METRICS
# ====================
volatility = float(portfolio_returns.std() * np.sqrt(252) * 100)

sharpe = 0.0 if portfolio_returns.std() == 0 else float(
    (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
)

beta_latest = float(rolling_beta.dropna().iloc[-1])

mkt_beta_latest = float(rolling_mkt_beta.dropna().iloc[-1])
mom_exp_latest = float(rolling_mom_corr.dropna().iloc[-1])

# ====================
# TABS
# ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Portfolio",
    "⚠️ Risk",
    "🚦 Signals",
    "🔄 Backtest"
])

# ====================
# TAB 1
# ====================
with tab1:

    st.subheader("Portfolio Engine")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Portfolio Return", f"{portfolio_total_return:.2f}%")
    col2.metric("SPY Return", f"{spy_total_return:.2f}%")
    col3.metric("Alpha vs SPY", f"{alpha:.2f}%")
    col4.metric("Sharpe", f"{sharpe:.2f}")
    st.metric("Rolling Beta (SPY)", f"{beta_latest:.2f}")
    st.metric("Market Beta", f"{mkt_beta_latest:.2f}")
    st.metric("Momentum Exposure", f"{mom_exp_latest:.2f}")
    st.metric("Optimized Sharpe", f"{opt_sharpe:.2f}")
    st.metric("Optimized Return", f"{opt_return * 100:.2f}%")
    st.metric("Optimized Volatility", f"{opt_vol * 100:.2f}%")

    st.plotly_chart(
        px.pie(allocation, names="Ticker", values="Weight", title="Allocation"),
        use_container_width=True
    )

    st.subheader("Optimized Portfolio (Max Sharpe)")

    opt_allocation = pd.DataFrame({
        "Ticker": ticker_list,
        "Weight": opt_weights
    })

    st.plotly_chart(
        px.pie(opt_allocation, names="Ticker", values="Weight", title="Optimized Allocation"),
        use_container_width=True
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cumulative.index,
        y=cumulative,
        name="Portfolio"
    ))

    fig.add_trace(go.Scatter(
        x=spy_cum.index,
        y=spy_cum,
        name="SPY Benchmark"
    ))

    fig_opt = go.Figure()

    fig_opt.add_trace(go.Scatter(
        x=cumulative.index,
        y=cumulative,
        name="Current Portfolio"
    ))

    fig_opt.add_trace(go.Scatter(
        x=optimized_cum.index,
        y=optimized_cum,
        name="Max Sharpe Portfolio"
    ))

    fig_opt.add_trace(go.Scatter(
        x=spy_cum.index,
        y=spy_cum,
        name="SPY"
    ))

    fig_opt.update_layout(
        title="Optimized vs Current vs SPY",
        yaxis_title="Growth of $1"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig_opt, use_container_width=True)

# ====================
# TAB 2 (RISK)
# ====================
with tab2:

    st.subheader("Risk Engine")

    st.plotly_chart(
        px.imshow(corr, text_auto=True, title="Correlation Matrix"),
        use_container_width=True
    )

    # ====================
    # DRAWDOWN
    # ====================
    fig_dd = go.Figure()

    fig_dd.add_trace(go.Scatter(
        x=cumulative.index,
        y=drawdown,
        name="Portfolio Drawdown"
    ))

    fig_dd.add_trace(go.Scatter(
        x=spy_drawdown.index,
        y=spy_drawdown,
        name="SPY Drawdown"
    ))

    fig_dd.update_layout(
        title="Drawdown Comparison",
        yaxis_title="Drawdown"
    )

    st.plotly_chart(fig_dd, use_container_width=True)

    # ====================
    # VOLATILITY
    # ====================
    fig_vol = go.Figure()

    fig_vol.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol,
        name="Portfolio Volatility"
    ))

    fig_vol.add_trace(go.Scatter(
        x=spy_vol.index,
        y=spy_vol,
        name="SPY Volatility"
    ))

    fig_vol.update_layout(
        title="Rolling Volatility (20D Annualized)",
        yaxis_title="Volatility"
    )

    st.plotly_chart(fig_vol, use_container_width=True)

    # ====================
    # ROLLING BETA
    # ====================
    fig_beta = go.Figure()

    fig_beta.add_trace(go.Scatter(
        x=rolling_beta.index,
        y=rolling_beta,
        name="Rolling Beta (20D)"
    ))

    fig_beta.add_hline(y=1.0, line_dash="dash", line_color="gray")

    fig_beta.update_layout(
        title="Rolling Beta vs SPY (20D)",
        yaxis_title="Beta"
    )

    st.plotly_chart(fig_beta, use_container_width=True)

    # ====================
    # MARKET FACTOR BETA
    # ====================
    # ====================
    # MOMENTUM EXPOSURE
    # ====================
    fig_mom = go.Figure()

    fig_mom.add_trace(go.Scatter(
        x=rolling_mom_corr.index,
        y=rolling_mom_corr,
        name="Momentum Exposure (Corr)"
    ))

    fig_mom.add_hline(y=0.0, line_dash="dash", line_color="gray")

    fig_mom.update_layout(
        title="Momentum Factor Exposure (Rolling)",
        yaxis_title="Correlation"
    )

    st.plotly_chart(fig_mom, use_container_width=True)

# ====================
# TAB 3
# ====================
with tab3:

    st.subheader("Signal Engine")

    st.metric("Portfolio Signal", final_signal)

    st.line_chart(portfolio_signal_series)

    st.dataframe(signal_df.tail(20))

# ====================
# TAB 4
# ====================
with tab4:

    st.subheader("Backtest Engine")

    backtest_df = pd.DataFrame({
        "Portfolio": cumulative,
        "SPY": spy_cum
    }).dropna()

    st.dataframe(backtest_df.tail(10))

    st.plotly_chart(
        go.Figure([
            go.Scatter(x=backtest_df.index, y=backtest_df["Portfolio"], name="Portfolio"),
            go.Scatter(x=backtest_df.index, y=backtest_df["SPY"], name="SPY")
        ]),
        use_container_width=True
    )

    st.plotly_chart(
        px.histogram(portfolio_returns, nbins=50),
        use_container_width=True
    )
