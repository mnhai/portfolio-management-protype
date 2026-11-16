import os
import py_compile
import pytest


def test_myapp0_exists():
    assert os.path.exists("myapp0.py"), "File myapp0.py must exist."


def test_myapp0_syntax():
    """Verify myapp0.py compiles without syntax errors."""
    try:
        py_compile.compile("myapp0.py", doraise=True)
    except py_compile.PyCompileError as e:
        pytest.fail(f"Syntax error in myapp0.py: {e}")


def test_requirements_exists():
    assert os.path.exists("requirements.txt")
