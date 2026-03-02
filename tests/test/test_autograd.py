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

    def test_article(self):
        """Check article example.

        Note that a is used twice (the graph branches), so its gradient is the sum of both paths.
        """
        from autograd import Value

        a = Value(2.0)
        b = Value(3.0)

        c = a * b
        assert c.data == 6.0
        loss = c + a
        assert loss.data == 8.0

        loss.backward()
        assert a.grad == 4.0  # dL/da = b + 1 = 3 + 1, via both paths
        assert b.grad == 2.0  # dL/db = a = 2
