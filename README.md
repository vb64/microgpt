# Hello World for LLM.

Forked from [karpathy/microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
Article [with explain](https://karpathy.github.io/2026/02/12/microgpt/).

Divided by modules, separate module for CLI. Data of trained model can be saved as json file and load from this file for model use.

## Setup

```sh
git clone git@github.com:vb64/microgpt.git
cd microgpt
make setup PYTHON_BIN=python3
```

## Usage

Train model on 50 documents from `fixtures/en_names.txt` dataset and saves data of the model to `build/model.json`.

```sh
make cmd_learn
```

Load data for model from `fixtures/en_names_1x16x16_doc_32032.json` (32 032 documents, 1 layer, 16 embedding dimension, 16 maximum context length of the attention window).

```sh
make cmd_ask
```

Then model generates names with different values of the temperature parameter.

```sh
MicroGPT runner.
Load model: fixtures/en_names_1x16x16_doc_32032.json
Parameters: 4192
# obabe (temperature=1)
# ayni (temperature=0.5)
# jarian (temperature=0.3)
# jarian (temperature=0.1)
# jarian (temperature=0.01)
# jarian (temperature=0.001)
# jarian (temperature=0.0001)
Done
```
