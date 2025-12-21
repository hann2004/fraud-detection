def test_ci_cd():
    """Simple test to verify CI/CD is working"""
    assert 1 + 1 == 2


def test_imports():
    """Test that key packages can be imported"""
    try:
        import pandas
        import numpy
        import sklearn
        assert True
    except ImportError:
        assert False, "Required packages not installed"