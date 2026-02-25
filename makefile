.PHONY: all setup
# make tests >debug.log 2>&1
ifeq ($(OS),Windows_NT)
PYTHON = venv/Scripts/python.exe
PTEST = venv/Scripts/pytest.exe
COVERAGE = venv/Scripts/coverage.exe
else
PYTHON = ./venv/bin/python
PTEST = ./venv/bin/pytest
COVERAGE = ./venv/bin/coverage
endif

PIP = $(PYTHON) -m pip install
SOURCE = source
TESTS = tests

PYTEST = $(PTEST) --cov=$(SOURCE) --cov-report term:skip-covered
LINT = $(PYTHON) -m pylint

all:

cmd_dset:
	$(PYTHON) $(SOURCE)/cli.py dataset dataset/en_names.txt

test:
	$(PTEST) -s $(TESTS)/test/$(T)

tests: flake8 pep257 lint
	$(PYTEST) --durations=5 $(TESTS)
	$(COVERAGE) html --skip-covered

pep257:
	$(PYTHON) -m pydocstyle --match='.*\.py' $(TESTS)/test
	$(PYTHON) -m pydocstyle $(SOURCE)

flake8:
	$(PYTHON) -m flake8 $(TESTS)
	$(PYTHON) -m flake8 $(SOURCE)

lint:
	$(LINT) $(TESTS)/test
	$(LINT) $(SOURCE)

setup: setup_python setup_pip

setup_pip:
	$(PIP) --upgrade pip
	$(PIP) -r $(TESTS)/requirements.txt

setup_python:
	$(PYTHON_BIN) -m venv ./venv
