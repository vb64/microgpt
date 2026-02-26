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

        tok = Tokenizer(data.docs)
        assert tok.size == 27
        assert tok.bos == 26
        assert ': 27' in str(tok)

        doc = data.docs[0]
        assert doc == "emma"

        tokens = tok.tokenize(doc)
        assert tokens == [26, 4, 12, 12, 0, 26]
