## Piano Hero: A discrete latent variable model for piano music

Piano Hero is a system for learning a low-dimensional discrete representation of piano music. By learning a bidirectional mapping into and out of this space, we can create a simple music interface (consisting of a few buttons) which controls the entire piano.

Piano Hero uses an encoder RNN to compress piano sequences (88 keys) into many fewer buttons (e.g. 8). A decoder RNN is responsible for converting the simpler sequences back to piano space

### Usage

First, [set up your development environment](/magenta#development-environment). Then, [convert some MIDI files into `NoteSequence` records]((/magenta/scripts/README.md) to build a dataset for Piano Hero.

To train a Piano Hero model, run the following:

```
bazel run //magenta/models/piano_hero:train -- \
  --dataset_fp=/tmp/piano_hero/chopin_train.tfrecord \
  --train_dir=/tmp/piano_hero/training_run
```

To evaluate a model while it is training, run the following:

```
bazel run //magenta/models/piano_hero:eval -- \
  --dataset_fp=/tmp/piano_hero/chopin_validation.tfrecord \
  --train_dir=/tmp/piano_hero/training_run \
  --eval_dir==/tmp/piano_hero/training_run/eval_validation
```
