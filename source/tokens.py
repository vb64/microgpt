"""Tokenizer."""


class Tokenizer:
    """Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back."""

    def __init__(self, docs):
        """Create tokens for given doc list."""
        # unique characters in the dataset become token ids 0..n-1
        self.uchars = sorted(set(''.join(docs)))
        self.bos = len(self.uchars)  # token id for a special Beginning of Sequence (BOS) token
        self.size = self.bos + 1  # total number of unique tokens, +1 is for BOS

    def __str__(self):
        """Return string representation."""
        return "vocab size: {}".format(self.size)

    def tokenize(self, doc):
        """Take single document, tokenize it, surround it with BOS special token on both sides."""
        return [self.bos] + [self.uchars.index(ch) for ch in doc] + [self.bos]

    def to_json(self):
        """Return json coded Tokenizer data."""
        return ''.join(self.uchars)

    @classmethod
    def from_json(cls, data):
        """Create instance from json coded Tokenizer data."""
        tok = cls([])
        tok.uchars = sorted(set(data))
        tok.bos = len(tok.uchars)
        tok.size = tok.bos + 1

        return tok
