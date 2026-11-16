"""Unit tests for myapp9.py Max Sharpe optimization engine and constraints."""
import os
import py_compile
import pandas as pd
import numpy as np
import pytest
from myapp9 import optimize_max_sharpe


def test_myapp9_exists_and_compiles():
    assert os.path.exists("myapp9.py"), "File myapp9.py must exist."
    try:
        py_compile.compile("myapp9.py", doraise=True)
    except py_compile.PyCompileError as e:
        pytest.fail(f"Compilation error in myapp9.py: {e}")


def test_optimization_constraints():
    """Verify that SLSQP output weights sum to 1.0 and respect bounds."""
    np.random.seed(42)
    daily_returns = pd.DataFrame(
        np.random.normal(0.001, 0.02, (252, 3)),
        columns=["AAPL", "MSFT", "NVDA"]
    )
    assets = ["AAPL", "MSFT", "NVDA"]
    max_w = 0.60
    opt = optimize_max_sharpe(daily_returns, assets, rf=0.02, max_weight=max_w)

    assert opt["success"] is True
    weights = np.array(list(opt["weights"].values()))

    # Sum of weights = 1.0
    assert pytest.approx(weights.sum(), 0.0001) == 1.0

    # No shorting: w >= 0
    assert (weights >= -1e-6).all()

    # Max weight per asset constraint
    assert (weights <= max_w + 1e-4).all()


def test_sharpe_ratio_superiority():
    """Verify that optimized Sharpe ratio is >= equal-weighted Sharpe ratio."""
    np.random.seed(42)
    daily_returns = pd.DataFrame(
        np.random.normal(0.001, 0.02, (252, 3)),
        columns=["AAPL", "MSFT", "NVDA"]
    )
    assets = ["AAPL", "MSFT", "NVDA"]
    opt = optimize_max_sharpe(daily_returns, assets, rf=0.02, max_weight=1.0)

    # Equal weighted
    eq_w = np.ones(3) / 3
    mean_ret = daily_returns.mean().values * 252
    cov_mat = daily_returns.cov().values * 252
    eq_ret = np.dot(eq_w, mean_ret)
    eq_vol = np.sqrt(np.dot(eq_w.T, np.dot(cov_mat, eq_w)))
    eq_sharpe = (eq_ret - 0.02) / eq_vol

    assert opt["sharpe_ratio"] >= eq_sharpe - 1e-4
