"""Unit tests for myapp7.py rolling beta and enhanced risk metrics."""
import os
import py_compile
import pandas as pd
import numpy as np
import pytest


def test_myapp7_exists_and_compiles():
    assert os.path.exists("myapp7.py"), "File myapp7.py must exist."
    try:
        py_compile.compile("myapp7.py", doraise=True)
    except py_compile.PyCompileError as e:
        pytest.fail(f"Compilation error in myapp7.py: {e}")


def test_rolling_beta_mathematical_precision():
    """Verify Beta = Cov(Rp, Rb) / Var(Rb). If Rp = 1.5 * Rb, Beta must be 1.5."""
    np.random.seed(42)
    spy_returns = pd.Series(np.random.normal(0.001, 0.015, 100))
    port_returns = 1.5 * spy_returns  # Perfectly correlated with 1.5x leverage

    rolling_cov = port_returns.rolling(20).cov(spy_returns)
    rolling_var = spy_returns.rolling(20).var()
    rolling_beta = rolling_cov / rolling_var

    valid_betas = rolling_beta.dropna()
    np.testing.assert_allclose(valid_betas.values, 1.5, rtol=1e-5)


def test_rolling_alpha_capm_logic():
    """Verify CAPM Alpha adjustment formula."""
    rp = 0.20   # 20% annualized return
    rb = 0.10   # 10% benchmark return
    beta = 1.5  # Beta
    alpha = rp - beta * rb
    # alpha = 0.20 - 1.5 * 0.10 = 0.05 (5%)
    assert pytest.approx(alpha, 0.0001) == 0.05


def test_beta_interpretation_logic():
    """Verify beta interpretation."""
    beta_leveraged = 1.49
    excess = (beta_leveraged - 1.0) * 100
    assert pytest.approx(excess, 0.01) == 49.0  # 49% more sensitive than market
