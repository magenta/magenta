## Drums RNN

This model applies language modeling to drum track generation using an LSTM. Unlike melodies, drum tracks are polyphonic in the sense that multiple drums can be struck simultaneously. Despite this, we model a drum track as a single sequence of events by a) mapping all of the different MIDI drums onto a smaller number of drum classes, and b) representing each event as a single value representing the set of drum classes that are struck.

## Configurations

### One Drum

This configuration maps all drums to a single drum class. It uses a basic binary encoding of drum tracks, where the value at each step is 0 if silence and 1 if at least one drum is struck.

### Drum Kit

This configuration maps all drums to a 9-piece drum kit consisting of bass drum, snare drum, closed and open hi-hat, three toms, and crash and ride cymbals. The set of drums is encoded as a length 512 one-hot vector, where each bit of the vector corresponds to one of the 9 drums. The input to the model is also augmented with binary counters.

## How to Use

First, set up your [Magenta environment](/README.md). Next, you can either use a pre-trained model or train your own.

## Pre-trained

If you want to get started right away, you can use a Drum Kit model that we've pre-trained on thousands of MIDI files:

* [drum_kit](http://download.magenta.tensorflow.org/models/drum_kit_rnn.mag)

### Generate a drum track

```
BUNDLE_PATH=<absolute path of .mag file>
CONFIG=<one of 'one_drum' or 'drum_kit', matching the bundle>

drums_rnn_generate \
--config=${CONFIG} \
--bundle_file=${BUNDLE_PATH} \
--output_dir=/tmp/drums_rnn/generated \
--num_outputs=10 \
--num_steps=128 \
--primer_drums="[(36,)]"
```

This will generate a drum track starting with a bass drum hit. If you'd like, you can also supply priming drums using a string representation of a Python list. The values in the list should be tuples of integer MIDI pitches representing the drums that are played simultaneously at each step. For example `--primer_drums="[(36, 42), (), (42,)]"` would prime the model with one step of bass drum and hi-hat, then one step of rest, then one step of just hi-hat. Instead of using `--primer_drums`, we can use `--primer_midi` to prime our model with drums stored in a MIDI file.

## Train your own

### Create NoteSequences

Our first step will be to convert a collection of MIDI files into NoteSequences. NoteSequences are [protocol buffers](https://developers.google.com/protocol-buffers/), which is a fast and efficient data format, and easier to work with than MIDI files. See [Building your Dataset](/magenta/scripts/README.md) for instructions on generating a TFRecord file of NoteSequences. In this example, we assume the NoteSequences were output to ```/tmp/notesequences.tfrecord```.

### Create SequenceExamples

SequenceExamples are fed into the model during training and evaluation. Each SequenceExample will contain a sequence of inputs and a sequence of labels that represent a drum track. Run the command below to extract drum tracks from our NoteSequences and save them as SequenceExamples. Two collections of SequenceExamples will be generated, one for training, and one for evaluation, where the fraction of SequenceExamples in the evaluation set is determined by `--eval_ratio`. With an eval ratio of 0.10, 10% of the extracted drum tracks will be saved in the eval collection, and 90% will be saved in the training collection.

```
drums_rnn_create_dataset \
--config=<one of 'one_drum' or 'drum_kit'> \
--input=/tmp/notesequences.tfrecord \
--output_dir=/tmp/drums_rnn/sequence_examples \
--eval_ratio=0.10
```

### Train and Evaluate the Model

Run the command below to start a training job using the attention configuration. `--run_dir` is the directory where checkpoints and TensorBoard data for this run will be stored. `--sequence_example_file` is the TFRecord file of SequenceExamples that will be fed to the model. `--num_training_steps` (optional) is how many update steps to take before exiting the training loop. If left unspecified, the training loop will run until terminated manually. `--hparams` (optional) can be used to specify hyperparameters other than the defaults. For this example, we specify a custom batch size of 64 instead of the default batch size of 128. Using smaller batch sizes can help reduce memory usage, which can resolve potential out-of-memory issues when training larger models. We'll also use a 2-layer RNN with 64 units each, instead of the default of 3 layers of 256 units each. This will make our model train faster. However, if you have enough compute power, you can try using larger layer sizes for better results. You can also adjust how many previous steps the attention mechanism looks at by changing the `attn_length` hyperparameter. For this example we leave it at the default value of 32 steps (2 bars).

```
drums_rnn_train \
--config=drum_kit \
--run_dir=/tmp/drums_rnn/logdir/run1 \
--sequence_example_file=/tmp/drums_rnn/sequence_examples/training_drum_tracks.tfrecord \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--num_training_steps=20000
```

Optionally run an eval job in parallel. `--run_dir`, `--hparams`, and `--num_training_steps` should all be the same values used for the training job. `--sequence_example_file` should point to the separate set of eval drum tracks. Include `--eval` to make this an eval job, resulting in the model only being evaluated without any of the weights being updated.

```
drums_rnn_train \
--config=drum_kit \
--run_dir=/tmp/drums_rnn/logdir/run1 \
--sequence_example_file=/tmp/drums_rnn/sequence_examples/eval_drum_tracks.tfrecord \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--num_training_steps=20000 \
--eval
```

Run TensorBoard to view the training and evaluation data.

```
tensorboard --logdir=/tmp/drums_rnn/logdir
```

Then go to [http://localhost:6006](http://localhost:6006) to view the TensorBoard dashboard.

### Generate Drum Tracks

Drum tracks can be generated during or after training. Run the command below to generate a set of drum tracks using the latest checkpoint file of your trained model.

`--run_dir` should be the same directory used for the training job. The `train` subdirectory within `--run_dir` is where the latest checkpoint file will be loaded from. For example, if we use `--run_dir=/tmp/drums_rnn/logdir/run1`. The most recent checkpoint file in `/tmp/drums_rnn/logdir/run1/train` will be used.

`--hparams` should be the same hyperparameters used for the training job, although some of them will be ignored, like the batch size.

`--output_dir` is where the generated MIDI files will be saved. `--num_outputs` is the number of drum tracks that will be generated. `--num_steps` is how long each melody will be in 16th steps (128 steps = 8 bars).

At least one note needs to be fed to the model before it can start generating consecutive notes. We can use `--primer_drums` to specify a priming drum track using a string representation of a Python list. The values in the list should be tuples of integer MIDI pitches representing the drums that are played simultaneously at each step. For example `--primer_drums="[(36, 42), (), (42,)]"` would prime the model with one step of bass drum and hi-hat, then one step of rest, then one step of just hi-hat. Instead of using `--primer_drums`, we can use `--primer_midi` to prime our model with drums stored in a MIDI file. If neither `--primer_drums` nor `--primer_midi` are specified, a single step of bass drum will be used as the primer, then the remaining steps will be generated by the model. In the example below we prime the drum track with `--primer_drums="[(36,)]"`, a single bass drum hit.


```
drums_rnn_generate \
--config=drum_kit \
--run_dir=/tmp/drums_rnn/logdir/run1 \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--output_dir=/tmp/drums_rnn/generated \
--num_outputs=10 \
--num_steps=128 \
--primer_drums="[(36,)]"
```

### Creating a Bundle File

The [bundle format](https://github.com/magenta/note-seq/blob/master/note_seq/protobuf/generator.proto)
is a convenient way of combining the model checkpoint, metagraph, and
some metadata about the model into a single file.

To generate a bundle, use the
[create_bundle_file](/magenta/models/shared/sequence_generator.py)
method within SequenceGenerator. Our generator script
supports a ```--save_generator_bundle``` flag that calls this method. Example:

```sh
drums_rnn_generate \
  --config=drum_kit \
  --run_dir=/tmp/drums_rnn/logdir/run1 \
  --hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
  --bundle_file=/tmp/drums_rnn.mag \
  --save_generator_bundle
```
