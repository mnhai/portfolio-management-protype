"""Regression tests ensuring cumulative feature preservation across all app versions."""
import py_compile
import pytest


def _read(filename: str) -> str:
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("filename", [
    "myapp7.py", "myapp8.py", "myapp9.py", "myapp10.py", "myapp13.py"
])
def test_all_upgraded_apps_compile(filename):
    py_compile.compile(filename, doraise=True)


def test_myapp7_preserves_myapp5_features():
    src = _read("myapp7.py")
    assert "fig_beta" in src
    assert "fig_dd" in src
    assert "fig_corr" in src
    assert "fig_sig" in src  # Price + MA5/MA20 chart
    assert "fig_hist" in src


def test_myapp8_has_3_momentum_charts_and_preserves_myapp7():
    src = _read("myapp8.py")
    assert "rolling_vol" in src       # 1. Rolling Volatility
    assert "rolling_beta" in src      # 2. Rolling Beta vs SPY
    assert "rolling_mom_corr" in src  # 3. Momentum Factor Exposure
    assert "fig_dd" in src            # Preserved Drawdown
    assert "fig_corr" in src          # Preserved Correlation Matrix
    assert "fig_sig" in src           # Preserved MA5/MA20 Price chart
    assert "fig_hist" in src          # Preserved Histogram


def test_myapp9_preserves_myapp8_features():
    src = _read("myapp9.py")
    assert "optimize_max_sharpe" in src
    assert "fig_beta" in src
    assert "fig_dd" in src
    assert "fig_corr" in src
    assert "fig_sig" in src
    assert "fig_hist" in src
    assert "rolling_vol" in src
    assert "rolling_mom_corr" in src


def test_myapp10_preserves_myapp9_features():
    src = _read("myapp10.py")
    assert "compute_market_regimes" in src
    assert "simulate_regime_switching_strategy" in src
    assert "optimize_max_sharpe" in src
    assert "fig_dd" in src
    assert "fig_corr" in src
    assert "fig_sig" in src
    assert "rolling_vol" in src
    assert "fig_map" in src


def test_myapp13_preserves_all_features():
    src = _read("myapp13.py")
    assert "mc_vols" in src
    assert "capm_alpha" in src
    assert "treynor_ratio" in src
    assert "active_attributions" in src
    assert "fig_beta_heat" in src
    assert "fig_sig" in src
    assert "fig_dd" in src
    assert "rolling_vol" in src
    assert "mom_corr" in src
