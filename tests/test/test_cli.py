"""Module cli.py tests.

make test T=test_cli.py
"""
from . import TestBase, MockModel


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

    def test_cmd_learn(self):
        """Check learn command."""
        import cli

        model = cli.Model
        cli.Model = MockModel

        self.options.command = cli.Command.Learn
        assert cli.main(self.options) is None

        self.options.learn_cycles = 1
        assert cli.main(self.options) is None

        cli.Model = model

    def test_progress_bar(self):
        """Check progress_bar function."""
        from cli import progress_bar

        assert progress_bar(10, 1, 'xxx') is None
