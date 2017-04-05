# NSynth Baseline

Use spectrogram convolutional autoencoders for the NSynth dataset.

## Structure

The root directory contains utility libraries and binaries as described in
BUILD.

### Binaries:

*   train
*   eval
*   interp
*   make_test_set
*   save_z

### Libraries:

*   reader
*   datasets
*   utils

Each binary has a respective borg config in `borg/` and a script for compiling
and running different configs in `scripts/`. Each script can be run locally or
on borg, just change the functions called in `__main__()` and run `python
scripts/run_(script).py` from the command line.

All the binaries loads modules from `models/` that return `train_op()` and
`eval_op()`. Each model then also loads a config module from
`scripts/(model)_configs/` which contains the specific model structure. New
model architectures can be specified by adding new configs, and/or changing the
hparams dict in the run script.
