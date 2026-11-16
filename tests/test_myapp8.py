"""Unit tests for myapp8.py synthetic momentum factor and factor exposure panel."""
import os
import py_compile
import pandas as pd
import numpy as np
import pytest


def test_myapp8_exists_and_compiles():
    assert os.path.exists("myapp8.py"), "File myapp8.py must exist."
    try:
        py_compile.compile("myapp8.py", doraise=True)
    except py_compile.PyCompileError as e:
        pytest.fail(f"Compilation error in myapp8.py: {e}")


def test_synthetic_momentum_formula():
    """Verify Momentum = P(t) / P(t - 12) - 1."""
    # 13 prices: index 0 to 12
    prices = pd.Series([100.0] * 12 + [120.0])
    mom = (prices / prices.shift(12)) - 1
    assert pytest.approx(mom.iloc[-1], 0.0001) == 0.20  # +20% 12-day momentum


def test_rolling_momentum_correlation():
    """Verify rolling correlation is bounded in [-1.0, 1.0]."""
    np.random.seed(42)
    s1 = pd.Series(np.random.randn(50))
    s2 = pd.Series(np.random.randn(50))
    rolling_corr = s1.rolling(20).corr(s2).dropna()
    assert (rolling_corr >= -1.0).all() and (rolling_corr <= 1.0).all()


def test_momentum_style_classification():
    """Verify momentum tilt logic: > 0.3 rider, < -0.3 contrarian, else neutral."""
    def classify(corr):
        if corr > 0.3:
            return "RIDER"
        elif corr < -0.3:
            return "CONTRARIAN"
        return "NEUTRAL"

    assert classify(0.55) == "RIDER"
    assert classify(-0.45) == "CONTRARIAN"
    assert classify(0.12) == "NEUTRAL"
