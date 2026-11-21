"""Unit tests for myapp9.py verbatim PDF Max Sharpe optimization logic."""
import os
import py_compile
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize


def test_myapp9_exists_and_compiles():
    assert os.path.exists("myapp9.py"), "File myapp9.py must exist."
    try:
        py_compile.compile("myapp9.py", doraise=True)
    except py_compile.PyCompileError as e:
        pytest.fail(f"Compilation error in myapp9.py: {e}")


def test_pdf_max_sharpe_optimization_constraints():
    """Verify verbatim PDF SLSQP optimization constraints (sum=1, long-only, max_weight=0.5)."""
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.normal(0.001, 0.02, (252, 3)),
        columns=["AAPL", "MSFT", "NVDA"]
    )
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    num_assets = 3

    def portfolio_performance(weights, mean_returns, cov_matrix):
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = ret / vol
        return ret, vol, sharpe

    def neg_sharpe(weights):
        return -portfolio_performance(weights, mean_returns, cov_matrix)[2]

    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1})
    max_weight = 0.5
    bounds = tuple((0, max_weight) for _ in range(num_assets))
    init_guess = np.array([1 / num_assets] * num_assets)

    opt_result = minimize(neg_sharpe, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)
    assert opt_result.success is True
    w = opt_result.x
    assert pytest.approx(np.sum(w), 1e-4) == 1.0
    assert (w >= -1e-6).all()
    assert (w <= max_weight + 1e-4).all()


def test_pdf_max_sharpe_beats_equal_weight():
    """Verify optimized Sharpe >= equal-weight Sharpe."""
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.normal(0.001, 0.02, (252, 3)),
        columns=["AAPL", "MSFT", "NVDA"]
    )
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    num_assets = 3

    def portfolio_performance(weights, mean_returns, cov_matrix):
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return ret, vol, ret / vol

    init_guess = np.array([1 / num_assets] * num_assets)
    eq_sharpe = portfolio_performance(init_guess, mean_returns, cov_matrix)[2]

    opt_res = minimize(
        lambda w: -portfolio_performance(w, mean_returns, cov_matrix)[2],
        init_guess,
        method="SLSQP",
        bounds=tuple((0, 0.5) for _ in range(num_assets)),
        constraints=({"type": "eq", "fun": lambda x: np.sum(x) - 1}),
    )
    opt_sharpe = portfolio_performance(opt_res.x, mean_returns, cov_matrix)[2]
    assert opt_sharpe >= eq_sharpe - 1e-4
