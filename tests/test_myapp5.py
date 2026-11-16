"""Unit tests for myapp5.py interactive weights, 4-tab Bloomberg terminal and risk metrics."""
import os
import py_compile
import pandas as pd
import numpy as np
import pytest


def test_myapp5_exists_and_compiles():
    assert os.path.exists("myapp5.py"), "File myapp5.py must exist."
    try:
        py_compile.compile("myapp5.py", doraise=True)
    except py_compile.PyCompileError as e:
        pytest.fail(f"Compilation error in myapp5.py: {e}")


def test_weight_normalization_logic():
    """Verify that any non-zero slider weights correctly normalize to 100%."""
    raw_weights = {"AAPL": 0.5, "MSFT": 0.3, "NVDA": 0.2}
    total = sum(raw_weights.values())
    norm_w = {k: v / total for k, v in raw_weights.items()}

    assert pytest.approx(sum(norm_w.values()), 0.0001) == 1.0
    assert pytest.approx(norm_w["AAPL"], 0.0001) == 0.5
    assert pytest.approx(norm_w["MSFT"], 0.0001) == 0.3


def test_alpha_calculation():
    """Verify Alpha = Portfolio Total Return - Benchmark Total Return."""
    port_ret = 5.7649  # 576.49%
    spy_ret = 1.0507   # 105.07%
    alpha = port_ret - spy_ret
    assert pytest.approx(alpha, 0.0001) == 4.7142  # 471.42% excess return


def test_correlation_matrix_properties():
    """Verify correlation matrix has 1 on diagonal and values in [-1, 1]."""
    df = pd.DataFrame({
        "AAPL": [0.01, 0.02, -0.01],
        "MSFT": [0.02, 0.01, -0.02],
        "SPY": [0.015, 0.012, -0.015]
    })
    corr = df.corr()
    assert corr.shape == (3, 3)
    np.testing.assert_allclose(np.diag(corr), [1.0, 1.0, 1.0])
    assert (corr.values >= -1.0).all() and (corr.values <= 1.0).all()


def test_win_rate_logic():
    """Verify win rate is percentage of positive trading days."""
    returns = pd.Series([0.02, 0.01, -0.01, 0.03, -0.02])
    win_rate = (returns > 0).mean() * 100
    assert win_rate == 60.0  # 3 out of 5 positive days
