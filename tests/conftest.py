"""Pytest session setup."""
import sys
import os
import pytest


def path_setup():
    """Setup sys.path."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(1, os.path.join(os.path.dirname(test_dir), 'source'))


@pytest.fixture(scope="session", autouse=True)
def session_setup(request):
    """Auto session resource fixture."""
    path_setup()
