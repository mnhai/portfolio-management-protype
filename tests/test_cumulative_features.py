"""Regression tests ensuring cumulative feature preservation and MD-spec compliance across all app versions."""
import py_compile
import pytest


def _read(filename: str) -> str:
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("filename", [
    "myapp1.py", "myapp2.py", "myapp5.py", "myapp7.py", "myapp8.py", "myapp9.py", "myapp10.py", "myapp13.py"
])
def test_all_apps_compile(filename):
    py_compile.compile(filename, doraise=True)


def test_myapp7_preserves_myapp5_features():
    src = _read("myapp7.py")
    assert "fig_beta" in src
    assert "fig_dd" in src
    assert "fig_corr" in src
    assert "fig_rvol" in src
    assert "fig_sig" in src
    assert "fig_hist" in src


def test_myapp8_has_5_risk_charts_and_preserves_myapp7():
    src = _read("myapp8.py")
    assert "roll_vol_port" in src
    assert "rolling_beta" in src
    assert "rolling_mom_corr" in src
    assert "fig_dd" in src
    assert "fig_corr" in src
    assert "fig_sig" in src
    assert "fig_hist" in src


def test_myapp9_preserves_myapp8_features():
    src = _read("myapp9.py")
    assert "optimize_max_sharpe" in src
    assert "fig_opt_pie" in src
    assert "fig_opt_comp" in src
    assert "fig_beta" in src
    assert "fig_dd" in src
    assert "fig_corr" in src
    assert "fig_sig" in src
    assert "fig_hist" in src
    assert "roll_vol_port" in src
    assert "rolling_mom_corr" in src


def test_myapp10_preserves_myapp9_features():
    src = _read("myapp10.py")
    assert "compute_market_regimes" in src
    assert "simulate_regime_switching_strategy" in src
    assert "optimize_max_sharpe" in src
    assert "fig_dd" in src
    assert "fig_corr" in src
    assert "fig_sig" in src
    assert "roll_vol_port" in src
    assert "fig_strat" in src
    assert "fig_score" in src


def test_myapp13_matches_final_md_spec():
    src = _read("myapp13.py")
    assert "mc_vols" in src
    assert "capm_alpha" in src
    assert "treynor_ratio" in src
    assert "cum_active_return" in src
    assert "fig_bheat" in src
    assert "fig_sig" in src
    assert "fig_dd" in src
    assert "roll_vol_port" in src
    assert "rolling_mom_corr" in src
    assert "fig_strat" in src
    assert "fig_map" in src
    assert "factor_betas" in src
    assert "beta_60d_matrix" in src
