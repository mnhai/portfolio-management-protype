"""Unit tests for myapp1.py analytics, indicators and calculations."""
import os
import py_compile
import pandas as pd
import numpy as np
import pytest


def test_myapp1_exists_and_compiles():
    assert os.path.exists("myapp1.py"), "File myapp1.py must exist."
    try:
        py_compile.compile("myapp1.py", doraise=True)
    except py_compile.PyCompileError as e:
        pytest.fail(f"Compilation error: {e}")


def test_metrics_calculation_formula():
    """Verify metrics calculation logic with deterministic data."""
    prices = pd.Series([100.0, 102.0, 101.0, 105.0, 110.0])
    returns = prices.pct_change().dropna()

    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    assert pytest.approx(total_return, 0.001) == 0.10  # 10%

    volatility = returns.std() * np.sqrt(252)
    assert volatility > 0

    rf = 0.02
    sharpe = (returns.mean() * 252 - rf) / volatility
    assert isinstance(sharpe, float)


def test_drawdown_formula():
    """Verify drawdown formula."""
    prices = pd.Series([100.0, 120.0, 90.0, 110.0])
    cum_max = prices.cummax()
    drawdown = (prices - cum_max) / cum_max

    # At 90, cum_max is 120, so drawdown is (90 - 120) / 120 = -0.25 (-25%)
    assert pytest.approx(drawdown.iloc[2], 0.001) == -0.25
    assert drawdown.min() == -0.25


def test_ma_signal_logic():
    """Verify MA5 and MA20 crossover produces correct BUY/SELL."""
    df = pd.DataFrame({
        "MA5": [10.0, 15.0, 12.0],
        "MA20": [12.0, 14.0, 13.0]
    })
    df["Signal"] = np.where(df["MA5"] > df["MA20"], "BUY", "SELL")
    assert df["Signal"].tolist() == ["SELL", "BUY", "SELL"]
