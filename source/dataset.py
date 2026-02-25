"""Dataset stuff."""
import argparse


class Dataset:
    """Dataset for training model."""

    def __init__(self, fname):
        """Load from given file."""
        with open(fname, "rt", encoding='utf-8') as inp:
            lines = inp.readlines()

        self.docs = [
          line.strip()
          for line in lines
          if line.strip()
        ]
        self.file_name = fname
