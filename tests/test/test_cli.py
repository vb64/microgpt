"""Module cli.py tests.

make test T=test_cli.py
"""
from . import TestBase


class TestMain(TestBase):
    """Tests console client."""

    def setUp(self):
        """Set options."""
        super().setUp()

        from cli import PARSER, Command
        self.options = PARSER.parse_args(args=[
          Command.Dataset,
          self.fixture('en_names.txt'),
        ])

    def test_cmd_dataset(self):
        """Check dataset command."""
        from cli import main

        assert main(self.options) is None

        self.options.command = 'not_exist'
        assert main(self.options) is None
