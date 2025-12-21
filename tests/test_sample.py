def test_ci_cd():
    """Simple test to verify CI/CD is working"""
    assert 1 + 1 == 2


def test_imports():
    """Test that key packages are available without importing them."""
    import importlib.util

    missing = []
    for pkg in ["pandas", "numpy", "sklearn"]:
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)

    if missing:
        raise AssertionError(f"Required packages not installed: {', '.join(missing)}")
