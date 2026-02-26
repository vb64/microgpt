"""Module autograd.py tests.

make test T=test_autograd.py
"""
from . import TestBase


class TestAutograd(TestBase):
    """Tests autograd.py module."""

    def test_value(self):
        """Check Value methods."""
        from autograd import Value

        # __rsub__
        assert Value(1) - 1
        assert 1 - Value(1)
        # __rtruediv__
        assert Value(1) / 1
        assert 1 / Value(1)
