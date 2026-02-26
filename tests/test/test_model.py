"""Module model.py tests.

make test T=test_model.py
"""
from . import TestBase


class TestModel(TestBase):
    """Tests model.py module."""

    def test_model(self):
        """Check model methods."""
        from model import Model

        model = Model()
        assert model.state_dict
        assert model.learn(self.data.docs[:3]) == 3584
