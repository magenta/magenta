# musicgen

This is our project repo within the bigger Magenta repo. All of our new code will go here, but we can also feel free to modify parts of Magenta as needed.

## Getting Started

First, follow the "Development Environment" steps in the [main Magenta README](../README.md) to get tensorflow/magenta installed.

The code in this sub-repo is divided into `common` and `experiments`.
`common` is structured as a Python package that contains all the code we expect to be common to all of our experiments (datasets, models, training code, sampling code, etc.)
Each subdirectory of `experiments` represents a new experiment, i.e. training a particular model on a particular dataset with particular options/hyperparams.

To make the `common` package available for import by code in `experiments`, you should first run
```bash
source setupenv.sh
```
with your current working directory as `/musicgen` (i.e. the sub-repo roo); this will add the sub-repo root to your `$PYTHONPATH` environment variable.
Do this every time you fire up a new terminal to run code in this sub-repo.

For consistency, all experiments should be run from the sub-repo root. To verify that you have everything set up correctly, make sure that you can run
```bash
python experiments/rnn_independent/train.py
```
