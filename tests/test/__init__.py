"""Base test suite module."""
import os
from unittest import TestCase


class MockModel:
    """Mocked GPT model."""

    def learn(self, docs, progress_bar=None):
        """Mock learn method."""
        return len(docs) + (1 if progress_bar else 0)

    def ask(self, temperature=1):
        """Mock ask method."""
        return "temperature{}".format(temperature)


class TestBase(TestCase):
    """Base test class."""

    def setUp(self):
        """Set options."""
        super().setUp()
        from dataset import Dataset

        self.data = Dataset(self.fixture('en_names.txt'))
        assert len(self.data.docs) == 32033

    @staticmethod
    def fixture(*path):
        """Return full path for file in 'fixtures' dir."""
        return os.path.join('dataset', *path)

    @staticmethod
    def build(*path):
        """Return full path for file in 'build' dir."""
        if not path:
            return 'build'
        return os.path.join('build', *path)
