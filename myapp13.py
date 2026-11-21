# streamlit run myapp13.py

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

# ====================
# INDIVIDUAL ASSET STATS
# ====================
asset_stats = pd.DataFrame({
    "Ticker": ticker_list,
    "Return": mean_returns.values,
    "Volatility": np.sqrt(np.diag(cov_matrix))
})


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
# MONTE CARLO PORTFOLIOS
# ====================
n_portfolios = 5000

mc_returns = []
mc_vols = []
mc_sharpes = []
mc_weights = []

for _ in range(n_portfolios):
    w = np.random.random(len(ticker_list))
    w /= w.sum()

    ret = np.sum(mean_returns * w)

    vol = np.sqrt(
        np.dot(w.T, np.dot(cov_matrix, w))
    )

    sharpe_mc = ret / vol if vol > 0 else 0

    mc_returns.append(ret)
    mc_vols.append(vol)
    mc_sharpes.append(sharpe_mc)
    mc_weights.append(w)

mc_df = pd.DataFrame({
    "Return": mc_returns,
    "Volatility": mc_vols,
    "Sharpe": mc_sharpes
})

max_sharpe_idx = mc_df["Sharpe"].idxmax()

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
# FACTOR DATA
# ====================
factor_prices = yf.download(
    ["SPY", "QQQ", "IWM", "^VIX"],
    start=start_date,
    auto_adjust=True
)["Close"]

factor_returns = factor_prices.pct_change().dropna()

# ====================
# FACTOR BETAS
# ====================
factor_betas = {}

for factor in ["SPY", "QQQ", "IWM", "^VIX"]:
    aligned_factor = pd.concat(
        [portfolio_returns, factor_returns[factor]],
        axis=1
    ).dropna()

    cov = aligned_factor.iloc[:, 0].cov(aligned_factor.iloc[:, 1])
    var = aligned_factor.iloc[:, 1].var()

    factor_betas[factor] = cov / var

factor_beta_df = pd.DataFrame(
    factor_betas.items(),
    columns=["Factor", "Beta"]
)

# ====================
# CAPM ALPHA
# ====================
market_beta = factor_betas["SPY"]

capm_alpha = (
    portfolio_returns.mean() * 252
    - market_beta * factor_returns["SPY"].mean() * 252
)

treynor = (
    portfolio_returns.mean() * 252
    / market_beta
)

# ====================
# REGIME SWITCHING ENGINE
# ====================
spy_price = spy

# trend signals
spy_ma50 = spy_price.rolling(50).mean()
spy_ma200 = spy_price.rolling(200).mean()

trend_signal = (spy_ma50 > spy_ma200).astype(int)

# momentum signal (12D)
spy_mom = spy_price.pct_change(12)
mom_signal = (spy_mom > 0).astype(int)

# volatility signal (risk-off when high vol)
spy_vol_regime = spy_returns.rolling(20).std() * np.sqrt(252)
vol_threshold = spy_vol_regime.quantile(0.7)

vol_signal = (spy_vol_regime < vol_threshold).astype(int)

# align
regime_df = pd.concat([trend_signal, mom_signal, vol_signal], axis=1).dropna()
regime_df.columns = ["trend", "momentum", "vol"]

regime_score = regime_df.mean(axis=1)

# final regime
regime_state = np.where(
    regime_score > 0.66, "BULL",
    np.where(regime_score < 0.33, "BEAR", "NEUTRAL")
)

spy_ma200 = factor_prices["SPY"].rolling(200).mean()
qqq_ma200 = factor_prices["QQQ"].rolling(200).mean()

risk_on = (
    (factor_prices["SPY"] > spy_ma200)
    & (factor_prices["QQQ"] > qqq_ma200)
    & (factor_prices["^VIX"] < 20)
)

current_regime = (
    "RISK ON"
    if risk_on.iloc[-1]
    else "RISK OFF"
)

# ====================
# ALIGN SPY WITH REGIME
# ====================
spy_df = pd.DataFrame({
    "SPY": spy_price.reindex(regime_df.index),
})

spy_df["Regime"] = regime_state[:len(spy_df)]
spy_df = spy_df.dropna()

# ====================
# REGIME ALLOCATION ENGINE
# ====================


def equal_weight(n):
    return np.array([1 / n] * n)


eq_weights = equal_weight(len(ticker_list))

# define allocations
bull_weights = opt_weights          # from optimizer
bear_weights = np.zeros(len(ticker_list))
# fallback defensive = SPY proxy (equal-weight proxy if needed)
neutral_weights = eq_weights

strategy_returns = []

for i in range(len(portfolio_aligned)):
    state = regime_state[i] if i < len(regime_state) else "NEUTRAL"
    if state == "BULL":
        w = bull_weights
    elif state == "BEAR":
        w = bear_weights
    else:
        w = neutral_weights

    r = np.dot(returns.iloc[i], w)
    strategy_returns.append(r)

strategy_returns = pd.Series(strategy_returns, index=portfolio_aligned.index)
strategy_cum = (1 + strategy_returns).cumprod()

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

# ====================
# ATTRIBUTION vs SPY
# ====================
active_returns = portfolio_aligned - spy_aligned
attribution = pd.DataFrame({
    "Active Return": active_returns
})

attribution["Cumulative Active"] = (1 + attribution["Active Return"]).cumprod()

factor_df.columns = ["portfolio", "market", "momentum"]

rolling_window = 60

beta_matrix = pd.DataFrame()

spy_factor = factor_returns["SPY"]

for ticker in ticker_list:
    aligned = pd.concat(
        [returns[ticker], spy_factor],
        axis=1
    ).dropna()

    beta = (
        aligned.iloc[:, 0]
        .rolling(rolling_window)
        .cov(aligned.iloc[:, 1])
        /
        aligned.iloc[:, 1]
        .rolling(rolling_window)
        .var()
    )

    beta_matrix[ticker] = beta

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Portfolio",
    "⚠️ Risk",
    "🚦 Signals",
    "🔄 Backtest",
    "📈 Factors"
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
    st.metric("CAPM Alpha", f"{capm_alpha * 100:.2f}%")
    st.metric("Treynor Ratio", f"{treynor:.2f}")
    st.metric("Regime", current_regime)

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

    # ----------------------
    # Individual Assets
    # ----------------------
    fig_opt.add_trace(
        go.Scatter(
            x=asset_stats["Volatility"],
            y=asset_stats["Return"],
            mode="markers+text",
            text=asset_stats["Ticker"],
            textposition="top center",
            marker=dict(
                size=14,
                symbol="diamond"
            ),
            name="Assets"
        )
    )

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

    # ====================
    # EFFICIENT FRONTIER
    # ====================
    fig_frontier = px.scatter(
        mc_df,
        x="Volatility",
        y="Return",
        color="Sharpe",
        title="Monte Carlo Efficient Frontier"
    )
    fig_frontier.update_traces(
        marker=dict(opacity=0.3)
    )

    # ----------------------
    # Individual Assets
    # ----------------------
    fig_frontier.add_trace(
        go.Scatter(
            x=asset_stats["Volatility"],
            y=asset_stats["Return"],
            mode="markers+text",
            text=asset_stats["Ticker"],
            textposition="top center",
            textfont=dict(
                color="black", size=14  # ~2x larger
            ),
            marker=dict(
                size=14,
                symbol="diamond",
                color="black",
                line=dict(width=1)
            ),
            name="Assets"
        )
    )

    # optimized portfolio
    fig_frontier.add_trace(
        go.Scatter(
            x=[opt_vol],
            y=[opt_return],
            mode="markers",
            marker=dict(
                size=32, color="red",
                symbol="star"
            ),
            name="Max Sharpe"
        )
    )

    fig_frontier.update_layout(
        coloraxis_colorbar=dict(
            title="Sharpe",
            x=1.15,  # move further right
            y=0.5,
            len=0.75
        )
    )

    st.plotly_chart(
        fig_frontier,
        use_container_width=True
    )

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

    st.subheader("Regime Switching Strategy")

    fig_reg = go.Figure()

    fig_reg.add_trace(go.Scatter(
        x=strategy_cum.index,
        y=strategy_cum,
        name="Regime Strategy"
    ))

    fig_reg.add_trace(go.Scatter(
        x=cumulative.index,
        y=cumulative,
        name="Static Portfolio"
    ))

    fig_reg.add_trace(go.Scatter(
        x=spy_cum.index,
        y=spy_cum,
        name="SPY"
    ))

    fig_reg.update_layout(
        title="Bull/Bear Regime Switching Strategy",
        yaxis_title="Growth of $1"
    )

    st.plotly_chart(fig_reg, use_container_width=True)

    regime_map = pd.DataFrame({
        "Date": regime_df.index,
        "Regime": regime_state[:len(regime_df)]
    })

    st.subheader("Market Regime")

    st.line_chart(regime_score)

    st.dataframe(regime_map.tail(20))

    # ====================
    # SPY PRICE COLORED BY REGIME
    # ====================
    fig_regime_scatter = px.scatter(
        spy_df,
        x=spy_df.index,
        y="SPY",
        color="Regime",
        title="SPY Price Regime Map (Bull / Neutral / Bear)",
        color_discrete_map={
            "BULL": "green",
            "NEUTRAL": "orange",
            "BEAR": "red"
        }
    )

    fig_regime_scatter.update_traces(mode="markers")

    fig_regime_scatter.update_layout(
        yaxis_title="SPY Price",
        xaxis_title="Date"
    )

    st.plotly_chart(fig_regime_scatter, use_container_width=True)

# ====================
# TAB 5 (FACTORS)
# ====================
with tab5:

    st.subheader("Factor Analytics")

    # ====================
    # FACTOR METRICS
    # ====================
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "SPY Beta",
        f"{factor_betas['SPY']:.2f}"
    )

    col2.metric(
        "QQQ Beta",
        f"{factor_betas['QQQ']:.2f}"
    )

    col3.metric(
        "IWM Beta",
        f"{factor_betas['IWM']:.2f}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "VIX Beta",
        f"{factor_betas['^VIX']:.2f}"
    )

    col2.metric(
        "CAPM Alpha",
        f"{capm_alpha * 100:.2f}%"
    )

    col3.metric(
        "Treynor Ratio",
        f"{treynor:.2f}"
    )

    st.metric(
        "Market Regime",
        current_regime
    )

    st.divider()

    # ====================
    # FACTOR BETA BAR CHART
    # ====================
    st.subheader("Factor Exposures")

    fig_factor = px.bar(
        factor_beta_df,
        x="Factor",
        y="Beta",
        text="Beta",
        title="Portfolio Factor Betas"
    )

    fig_factor.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig_factor,
        use_container_width=True
    )

    st.divider()

    # ====================
    # ROLLING BETA HEATMAP
    # ====================
    st.subheader("Rolling Beta Heatmap vs SPY")

    fig_heat = px.imshow(
        beta_matrix.T,
        aspect="auto",
        title="60-Day Rolling Beta"
    )

    st.plotly_chart(
        fig_heat,
        use_container_width=True
    )

    st.divider()

    # ====================
    # PORTFOLIO ATTRIBUTION
    # ====================
    st.subheader("Active Return Attribution vs SPY")

    fig_attr = go.Figure()

    fig_attr.add_trace(
        go.Scatter(
            x=attribution.index,
            y=attribution["Cumulative Active"],
            name="Active Return"
        )
    )

    fig_attr.update_layout(
        title="Portfolio Outperformance vs SPY",
        yaxis_title="Cumulative Active Return"
    )

    st.plotly_chart(
        fig_attr,
        use_container_width=True
    )

    st.divider()

    # ====================
    # FACTOR TABLE
    # ====================
    st.subheader("Factor Summary")

    factor_summary = pd.DataFrame({
        "Factor": ["SPY", "QQQ", "IWM", "VIX"],
        "Beta": [
            factor_betas["SPY"],
            factor_betas["QQQ"],
            factor_betas["IWM"],
            factor_betas["^VIX"]
        ]
    })

    st.dataframe(
        factor_summary,
        use_container_width=True
    )

    st.divider()

    # ====================
    # TOP/BOTTOM BETA TICKERS
    # ====================
    latest_betas = (
        beta_matrix
        .ffill()
        .iloc[-1]
        .sort_values(ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Highest Beta Names")
        st.dataframe(
            latest_betas.head(5).to_frame("Beta")
        )

    with col2:
        st.subheader("Lowest Beta Names")
        st.dataframe(
            latest_betas.tail(5).to_frame("Beta")
        )
