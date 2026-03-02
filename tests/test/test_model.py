"""Module model.py tests.

make test T=test_model.py
"""
from . import TestBase


class TestModel(TestBase):
    """Tests model.py module."""

    def test_model(self):
        """Check model methods."""
        from model import Model, random
        from autograd import Value

        random.seed(13)  # Let there be order among chaos. 42 ???

        model = Model()
        assert len(model.state_dict) == 7
        params = [p for mat in model.state_dict.values() for row in mat for p in row]
        assert len(params) == 3328

        returns = [True, False]

        assert model.learn(
          self.data.docs[:3],
          progress_bar=lambda total, step, txt: returns.pop()
        ) == 3584

        assert len(model.state_dict) == 9
        params = [p for mat in model.state_dict.values() for row in mat for p in row]
        assert len(params) == 3584

        param = params[0]
        assert isinstance(param, Value)

        assert round(param.data, 3) == -0.023
        assert param.grad == 0
        assert param._children == ()  # pylint: disable=protected-access
        assert param._local_grads == ()  # pylint: disable=protected-access

        model = Model()
        assert model.learn(self.data.docs[:3]) == 3584
        assert len(model.ask()) > 0

        model.block_size = 1
        assert len(model.ask()) > 0

        assert model.save(self.build("saved.json")) is None
