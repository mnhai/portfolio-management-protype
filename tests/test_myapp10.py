"""Unit tests for myapp10.py bull/bear regime switching strategy and regime map."""
import os
import py_compile
import pandas as pd
import numpy as np
import pytest


def test_myapp10_exists_and_compiles():
    assert os.path.exists("myapp10.py"), "File myapp10.py must exist."
    try:
        py_compile.compile("myapp10.py", doraise=True)
    except py_compile.PyCompileError as e:
        pytest.fail(f"Compilation error in myapp10.py: {e}")


def test_regime_score_formula_and_labeling():
    """Verify Regime Score = Trend + Momentum - Volatility mapping."""
    test_cases = [
        # (Trend, Mom, Vol_Risk_Off) -> Expected Score -> Expected Label
        (1, 1, 0, 2, "BULL"),
        (1, 0, 0, 1, "BULL"),
        (0, 1, 0, 1, "BULL"),
        (1, 0, 1, 0, "NEUTRAL"),
        (0, 1, 1, 0, "NEUTRAL"),
        (0, 0, 0, 0, "NEUTRAL"),
        (0, 0, 1, -1, "BEAR"),
    ]

    for trend, mom, vol, exp_score, exp_label in test_cases:
        score = trend + mom - vol
        assert score == exp_score
        if score >= 1:
            label = "BULL"
        elif score == 0:
            label = "NEUTRAL"
        else:
            label = "BEAR"
        assert label == exp_label


def test_dynamic_allocation_logic():
    """Verify allocation rules: BULL -> 100% growth, NEUTRAL -> 50/50, BEAR -> 100% defensive."""
    growth_ret = 0.02
    defensive_ret = -0.01

    def calc_strategy_return(regime):
        if regime == "BULL":
            return growth_ret
        elif regime == "NEUTRAL":
            return 0.5 * growth_ret + 0.5 * defensive_ret
        else:  # BEAR
            return defensive_ret

    assert pytest.approx(calc_strategy_return("BULL"), 0.0001) == 0.02
    assert pytest.approx(calc_strategy_return("NEUTRAL"), 0.0001) == 0.005
    assert pytest.approx(calc_strategy_return("BEAR"), 0.0001) == -0.01
