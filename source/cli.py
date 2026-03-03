"""CLI interface."""
import argparse

from model import Model
from dataset import Dataset


class Command:
    """Runner commands."""

    Dataset = "dataset"
    Learn = "learn"
    Ask = "ask"


PARSER = argparse.ArgumentParser(description='MicroGPT runner.')

PARSER.add_argument(
  "command",
  choices=[Command.Dataset, Command.Learn, Command.Ask],
  help="Command to execute."
)
PARSER.add_argument(
  "filename",
  help="Dataset or json file name."
)
PARSER.add_argument(
  "--learn_cycles",
  type=int,
  default=0,
  help="Number of learn cycles (integer). Default is 0 - entire dataset content."
)
PARSER.add_argument(
  "--save",
  default='',
  help="Name of json file for save model after learning. By default the model is not saved."
)


def progress_bar(_total, step, text):
    """Dump progress of model learning."""
    print("{} {}".format(step, text))


def main(options):
    """Entry point."""
    print("MicroGPT runner.")

    if options.command == Command.Dataset:
        data = Dataset(options.filename)
        print("Dataset from", data.file_name, "docs:", len(data.docs))

    elif options.command == Command.Learn:
        data = Dataset(options.filename)
        learn_cycles = len(data.docs)
        if options.learn_cycles:
            learn_cycles = min(options.learn_cycles, learn_cycles)
        print("Learn {} docs from dataset {}".format(learn_cycles, data.file_name))
        model = Model()
        parameters_count = model.learn(data.docs[:learn_cycles], progress_bar=progress_bar)
        print("Parameters:", parameters_count)
        if options.save:
            model.save(options.save)
            print("Save to:", options.save)

    elif options.command == Command.Ask:
        print("Load model:", options.filename)
        model = Model()
        parameters_count = model.load(options.filename)
        print("Parameters:", parameters_count)

    print("Done")


if __name__ == '__main__':  # pragma: no cover
    main(PARSER.parse_args())
