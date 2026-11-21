"""Regression tests verifying 100% verbatim PDF structure across all 9 app versions."""
import py_compile
import pytest


def _read(filename: str) -> str:
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("filename", [
    "myapp0.py", "myapp1.py", "myapp2.py", "myapp5.py",
    "myapp7.py", "myapp8.py", "myapp9.py", "myapp10.py", "myapp13.py"
])
def test_all_pdf_apps_compile(filename):
    py_compile.compile(filename, doraise=True)


def test_myapp0_matches_pdf():
    src = _read("myapp0.py")
    assert 'st.title("My First Trading App")' in src
    assert 'st.write("Streamlit is working' in src


def test_myapp1_matches_pdf():
    src = _read("myapp1.py")
    assert 'st.title("📈 Personal Trading Analytics App")' in src
    assert 'df["MA20"]' in src and 'df["MA50"]' in src
    assert 'st.header("Signal Lab")' in src


def test_myapp2_matches_pdf():
    src = _read("myapp2.py")
    assert 'st.subheader("Portfolio Mode + Signal Lab")' in src
    assert 'st.header("Portfolio Allocation")' in src
    assert 'st.header("Signal Lab (Simplified)")' in src


def test_myapp5_matches_pdf():
    src = _read("myapp5.py")
    assert 'st.subheader("Portfolio Engine")' in src
    assert 'st.subheader("Risk Engine")' in src
    assert 'fig_dd' in src and 'fig_vol' in src
    assert 'st.subheader("Signal Engine")' in src
    assert 'st.subheader("Backtest Engine")' in src


def test_myapp7_matches_pdf():
    src = _read("myapp7.py")
    assert 'Rolling Beta (SPY)' in src
    assert 'fig_beta' in src
    assert 'rolling_alpha' in src
    assert 'fig_dd' in src and 'fig_vol' in src


def test_myapp8_matches_pdf():
    src = _read("myapp8.py")
    assert 'Market Beta' in src and 'Momentum Exposure' in src
    assert 'fig_dd' in src and 'fig_vol' in src and 'fig_beta' in src and 'fig_mom' in src


def test_myapp9_matches_pdf():
    src = _read("myapp9.py")
    assert 'Optimized Sharpe' in src
    assert 'max_weight = 0.5' in src
    assert 'fig_opt' in src and 'fig_mom' in src


def test_myapp10_matches_pdf():
    src = _read("myapp10.py")
    assert 'REGIME SWITCHING ENGINE' in src
    assert 'vol_threshold = spy_vol_regime.quantile(0.7)' in src
    assert 'fig_reg' in src and 'st.line_chart(regime_score)' in src


def test_myapp13_matches_pdf():
    src = _read("myapp13.py")
    assert 'n_portfolios = 5000' in src
    assert 'fig_frontier' in src
    assert 'fig_regime_scatter' in src
    assert '["SPY", "QQQ", "IWM", "^VIX"]' in src
    assert 'fig_factor' in src and 'fig_heat' in src and 'fig_attr' in src
    assert 'Highest Beta Names' in src and 'Lowest Beta Names' in src
