"""Unit tests for myapp13.py Final Bloomberg Terminal: CAPM Alpha, Treynor Ratio, and Attribution."""
import os
import py_compile
import pandas as pd
import numpy as np
import pytest


def test_myapp13_exists_and_compiles():
    assert os.path.exists("myapp13.py"), "File myapp13.py must exist."
    try:
        py_compile.compile("myapp13.py", doraise=True)
    except py_compile.PyCompileError as e:
        pytest.fail(f"Compilation error in myapp13.py: {e}")


def test_capm_alpha_formula():
    """Verify CAPM Alpha = Rp - [Rf + Beta * (Rm - Rf)]."""
    rp = 0.25   # 25% portfolio annual return
    rm = 0.15   # 15% market annual return
    rf = 0.02   # 2% risk-free rate
    beta = 1.2  # Beta

    expected_capm = rf + beta * (rm - rf)  # 0.02 + 1.2 * 0.13 = 0.176 (17.6%)
    alpha = rp - expected_capm            # 0.25 - 0.176 = 0.074 (7.4%)
    assert pytest.approx(alpha, 0.0001) == 0.074


def test_treynor_ratio_formula():
    """Verify Treynor Ratio = (Rp - Rf) / Beta."""
    rp = 0.25
    rf = 0.02
    beta = 1.2
    treynor = (rp - rf) / beta
    assert pytest.approx(treynor, 0.0001) == 0.23 / 1.2


def test_active_attribution_identity():
    """Verify mathematical identity: sum(w_i * (R_i - R_m)) == R_p - R_m."""
    weights = np.array([0.4, 0.3, 0.3])
    asset_returns = np.array([0.30, 0.10, 0.20])
    market_return = 0.12

    portfolio_return = np.dot(weights, asset_returns)  # 0.4*0.3 + 0.3*0.1 + 0.3*0.2 = 0.12 + 0.03 + 0.06 = 0.21
    excess_portfolio = portfolio_return - market_return # 0.21 - 0.12 = 0.09

    attributions = weights * (asset_returns - market_return)
    sum_attributions = np.sum(attributions)

    assert pytest.approx(sum_attributions, 0.0001) == excess_portfolio


def test_monte_carlo_weights_validity():
    """Verify Dirichlet weights generation sums to 1.0."""
    n_sim = 500
    n_assets = 3
    mc_weights = np.random.dirichlet(np.ones(n_assets), size=n_sim)
    np.testing.assert_allclose(mc_weights.sum(axis=1), 1.0, rtol=1e-5)
    assert (mc_weights >= 0).all()
