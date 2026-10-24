"""Test helpers shared by selftest.py and the tests (pytest-compatible)."""


def slow(fn):
    """Mark a test as slow: selftest.py runs it only with --full."""
    fn.slow = True
    return fn
