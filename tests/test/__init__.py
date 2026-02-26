"""Base test suite module."""
import os
from unittest import TestCase


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
