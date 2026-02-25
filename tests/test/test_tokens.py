"""Module tokens.py tests.

make test T=test_tokens.py
"""
from . import TestBase


class TestTokens(TestBase):
    """Tests tokens.py module."""

    def test_cmd_dataset(self):
        """Check dataset command."""
        from dataset import Dataset
        from tokens import Tokenizer

        data = Dataset(self.fixture('en_names.txt'))
        assert len(data.docs) == 32033

        tokens = Tokenizer(data)
        assert tokens.size == 27
        assert tokens.bos == 26
        assert ': 27' in str(tokens)
