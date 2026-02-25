"""CLI interface."""
import argparse


class Command:
    """Runner commands."""

    Dataset = "dataset"


PARSER = argparse.ArgumentParser(description='MicroGPT runner.')

PARSER.add_argument(
  "command",
  choices=[Command.Dataset, ],
  help="Command to execute."
)
PARSER.add_argument(
  "filename",
  help="Dataset file name."
)

def main(options):
    """Entry point."""
    print("MicroGPT runner.")

    if options.command == Command.Dataset:
        print("Dataset from", options.filename)

    print("Done")


if __name__ == '__main__':  # pragma: no cover
    main(PARSER.parse_args())
