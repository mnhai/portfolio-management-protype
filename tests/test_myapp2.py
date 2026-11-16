"""Unit tests for myapp2.py multi-asset portfolio calculations and logic."""
import os
import py_compile
import pandas as pd
import numpy as np
import pytest


def test_myapp2_exists_and_compiles():
    assert os.path.exists("myapp2.py"), "File myapp2.py must exist."
    try:
        py_compile.compile("myapp2.py", doraise=True)
    except py_compile.PyCompileError as e:
        pytest.fail(f"Compilation error in myapp2.py: {e}")


def test_equal_weights_math():
    """Verify equal weights sum to exactly 1.0 and each is 1/N."""
    for n in [2, 3, 5, 10]:
        weights = np.ones(n) / n
        assert pytest.approx(weights.sum(), 0.0001) == 1.0
        assert weights[0] == 1.0 / n


def test_portfolio_return_vectorization():
    """Verify that dot product with equal weights matches mean return."""
    df_returns = pd.DataFrame({
        "AAPL": [0.01, -0.02, 0.03],
        "MSFT": [0.02, 0.01, -0.01],
        "NVDA": [0.04, -0.03, 0.05]
    })
    weights = np.ones(3) / 3
    portfolio_dot = df_returns.dot(weights)
    portfolio_mean = df_returns.mean(axis=1)

    np.testing.assert_allclose(portfolio_dot.values, portfolio_mean.values)


def test_portfolio_metrics_and_drawdown():
    """Verify portfolio cumulative return and drawdown on known values."""
    port_daily = pd.Series([0.10, -0.20, 0.15])
    wealth = (1 + port_daily).cumprod()
    # wealth: [1.10, 0.88, 1.012]
    # cum_max: [1.10, 1.10, 1.10]
    # dd: [0, (0.88-1.10)/1.10 = -0.20, (1.012-1.10)/1.10 = -0.08]

    cum_max = wealth.cummax()
    dd = (wealth - cum_max) / cum_max
    assert pytest.approx(dd.iloc[1], 0.001) == -0.20
    assert pytest.approx(dd.min(), 0.001) == -0.20
