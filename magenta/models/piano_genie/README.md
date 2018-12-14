## Piano Genie: A discrete latent variable model for piano music

Piano Genie is a system for learning a low-dimensional discrete representation of piano music. By learning a bidirectional mapping into and out of this space, we can create a simple music interface (consisting of a few buttons) which controls the entire piano.

Piano Genie uses an encoder RNN to compress piano sequences (88 keys) into many fewer buttons (e.g. 8). A decoder RNN is responsible for converting the simpler sequences back to piano space

### Usage

First, [set up your development environment](/magenta#development-environment). Then, [convert some MIDI files into `NoteSequence` records](/magenta/scripts/README.md) to build a dataset for Piano Genie.

To train a Piano Genie model, run the following:

```bash
python //magenta/models/piano_genie/train.py \
  --dataset_fp=/tmp/piano_genie/chopin_train.tfrecord \
  --train_dir=/tmp/piano_genie/training_run
```

To evaluate a model while it is training, run the following:

```bash
python magenta/models/piano_genie/eval.py \
  --dataset_fp=/tmp/piano_genie/chopin_validation.tfrecord \
  --train_dir=/tmp/piano_genie/training_run \
  --eval_dir==/tmp/piano_genie/training_run/eval_validation
```
