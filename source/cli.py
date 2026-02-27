"""CLI interface."""
import argparse

from model import Model
from dataset import Dataset


class Command:
    """Runner commands."""

    Dataset = "dataset"
    Learn = "learn"


PARSER = argparse.ArgumentParser(description='MicroGPT runner.')

PARSER.add_argument(
  "command",
  choices=[Command.Dataset, Command.Learn],
  help="Command to execute."
)
PARSER.add_argument(
  "filename",
  help="Dataset file name."
)
PARSER.add_argument(
  "--learn_cycles",
  type=int,
  default=0,
  help="Number of learn cycles (integer). Default is 0 - entire dataset content."
)


def progress_bar(total, step, text):
    """Dump progress of model learning."""
    print("{} -> {} {}".format(step, total, text))


def main(options):
    """Entry point."""
    print("MicroGPT runner.")
    data = Dataset(options.filename)

    if options.command == Command.Dataset:
        print("Dataset from", data.file_name, "docs:", len(data.docs))

    elif options.command == Command.Learn:
        learn_cycles = len(data.docs)
        if options.learn_cycles:
            learn_cycles = max(options.learn_cycles, learn_cycles)
        print("Learn {} docs from dataset {}".format(learn_cycles, data.file_name))
        model = Model()
        parameters_count = model.learn(data.docs[:learn_cycles], progress_bar=progress_bar)
        print("Parameters:", parameters_count)

    print("Done")


if __name__ == '__main__':  # pragma: no cover
    main(PARSER.parse_args())
