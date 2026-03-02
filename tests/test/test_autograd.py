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

        # Autograd calculated that if `loss` = a*b + a, and a=2 and b=3, then a.grad = 4.0
        # is telling us about the local influence of a on `loss`.
        # If you wiggle the inmput a, in what direction is `loss` changing?
        # Here, the derivative of `loss` w.r.t. a is 4.0, meaning that if we increase a by a tiny amount
        # (say 0.001), `loss` would increase by about 4x that (0.004).
        # Similarly, b.grad = 2.0 means the same nudge to b would increase `loss` by about 2x that (0.002).
        # In other words, these gradients tell us the direction (positive or negative depending on the sign),
        # and the steepness (the magnitude) of the influence of each individual input
        # on the final output (the loss).
        # This then allows us to interately nudge the parameters of our neural network to lower the loss,
        # and hence improve its predictions.
        loss.backward()
        assert a.grad == 4.0  # dL/da = b + 1 = 3 + 1, via both paths
        assert b.grad == 2.0  # dL/db = a = 2
