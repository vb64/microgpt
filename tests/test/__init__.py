"""Base test suite module."""
import os
from unittest import TestCase


class TestBase(TestCase):
    """Base test class."""

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
