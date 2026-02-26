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

        returns = [True, False]

        assert model.learn(
          self.data.docs[:3],
          progress_bar=lambda step, txt: returns.pop()
        ) == 3584

        model = Model()
        assert model.learn(self.data.docs[:3]) == 3584
        assert len(model.ask()) > 1
