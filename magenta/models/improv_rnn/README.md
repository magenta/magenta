## Improv RNN

This model generates melodies a la [Melody RNN](/magenta/models/melody_rnn/README.md), but conditions the melodies on an underlying chord progression. At each step of generation, the model is also given the current chord as input (encoded as a vector). Instead of training on MIDI files, the model is trained on lead sheets in MusicXML format.

## Configurations

### Basic Improv

This configuration is similar to the basic Melody RNN, but also provides the current chord encoded as a one-hot vector of 48 triads (major/minor/augmented/diminished for all 12 root pitch classes).

### Attention Improv

This configuration is similar to the attention Melody RNN, but also provides the current chord encoded as a one-hot vector of the 48 triads.

### Chord Pitches Improv

This configuration is similar to Basic Improv, but instead of using a one-hot encoding for chord triads, encodes a chord as the concatenation of the following length-12 vectors:

* a one-hot encoding of the chord root pitch class, e.g. `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]` for a D major (or minor, etc.) chord
* a binary vector indicating presence or absence of each pitch class, e.g. `[1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0]` for a C7#9 chord
* a one-hot encoding of the chord bass pitch class, which is usually the same as the chord root pitch class except in the case of "slash chords" like C/E

## How to Use

First, set up your [Magenta environment](/README.md). Next, you can either use a pre-trained model or train your own.

## Pre-trained

If you want to get started right away, you can use a model that we've pre-trained on thousands of MIDI files:

* [chord_pitches_improv](http://download.magenta.tensorflow.org/models/chord_pitches_improv.mag)

### Generate a melody over chords

```
BUNDLE_PATH=<absolute path of .mag file>
CONFIG=<one of 'basic_improv', 'attention_improv' or 'chord_pitches_improv', matching the bundle>

improv_rnn_generate \
--config=${CONFIG} \
--bundle_file=${BUNDLE_PATH} \
--output_dir=/tmp/improv_rnn/generated \
--num_outputs=10 \
--primer_melody="[60]" \
--backing_chords="C G Am F C G Am F" \
--render_chords
```

This will generate a melody starting with a middle C over the chord progression C G Am F, where each chord lasts one bar and the progression is repeated twice. If you'd like, you can supply a longer priming melody using a string representation of a Python list. The values in the list should be ints that follow the melodies_lib.Melody format (-2 = no event, -1 = note-off event, values 0 through 127 = note-on event for that MIDI pitch). For example `--primer_melody="[60, -2, 60, -2, 67, -2, 67, -2]"` would prime the model with the first four notes of *Twinkle Twinkle Little Star*. Instead of using `--primer_melody`, we can use `--primer_midi` to prime our model with a melody stored in a MIDI file. For example, `--primer_midi=<absolute path to magenta/models/melody_rnn/primer.mid>` will prime the model with the melody in that MIDI file.

You can modify the backing chords as you like; Magenta understands most basic chord types e.g. "A13", "Cdim", "F#m7b5". The `--steps_per_chord` option can be used to control the chord duration.

## Train your own

### Create NoteSequences

The Improv RNN trains on [lead sheets](https://en.wikipedia.org/wiki/Lead_sheet), a musical representation containing chords and melody (and lyrics, which are ignored by the model). You can find lead sheets in various places on the web such as [MuseScore](https://musescore.com). Magenta is currently only able to read lead sheets in MusicXML format; MuseScore provides MusicXML download links, e.g. [https://musescore.com/score/2779326/download/mxl].

Our first step will be to convert a collection of MusicXML lead sheets into NoteSequences. NoteSequences are [protocol buffers](https://developers.google.com/protocol-buffers/), which is a fast and efficient data format, and easier to work with than MIDI or MusicXML files. See [Building your Dataset](/magenta/scripts/README.md) for instructions on generating a TFRecord file of NoteSequences. In this example, we assume the NoteSequences were output to ```/tmp/notesequences.tfrecord```.

### Create SequenceExamples

SequenceExamples are fed into the model during training and evaluation. Each SequenceExample will contain a sequence of inputs and a sequence of labels that represent a lead sheet. Run the command below to extract lead sheets from our NoteSequences and save them as SequenceExamples. Two collections of SequenceExamples will be generated, one for training, and one for evaluation, where the fraction of SequenceExamples in the evaluation set is determined by `--eval_ratio`. With an eval ratio of 0.10, 10% of the extracted drum tracks will be saved in the eval collection, and 90% will be saved in the training collection.

```
improv_rnn_create_dataset \
--config=<one of 'basic_improv', 'attention_improv', or 'chord_pitches_improv'>
--input=/tmp/notesequences.tfrecord \
--output_dir=/tmp/improv_rnn/sequence_examples \
--eval_ratio=0.10
```

### Train and Evaluate the Model

Run the command below to start a training job using the attention configuration. `--run_dir` is the directory where checkpoints and TensorBoard data for this run will be stored. `--sequence_example_file` is the TFRecord file of SequenceExamples that will be fed to the model. `--num_training_steps` (optional) is how many update steps to take before exiting the training loop. If left unspecified, the training loop will run until terminated manually. `--hparams` (optional) can be used to specify hyperparameters other than the defaults. For this example, we specify a custom batch size of 64 instead of the default batch size of 128. Using smaller batch sizes can help reduce memory usage, which can resolve potential out-of-memory issues when training larger models. We'll also use a 2 layer RNN with 64 units each, instead of the default of 2 layers of 128 units each. This will make our model train faster. However, if you have enough compute power, you can try using larger layer sizes for better results. You can also adjust how many previous steps the attention mechanism looks at by changing the `attn_length` hyperparameter. For this example we leave it at the default value of 40 steps (2.5 bars).

```
improv_rnn_train \
--config=attention_improv \
--run_dir=/tmp/improv_rnn/logdir/run1 \
--sequence_example_file=/tmp/improv_rnn/sequence_examples/training_lead_sheets.tfrecord \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--num_training_steps=20000
```

Optionally run an eval job in parallel. `--run_dir`, `--hparams`, and `--num_training_steps` should all be the same values used for the training job. `--sequence_example_file` should point to the separate set of eval lead sheets. Include `--eval` to make this an eval job, resulting in the model only being evaluated without any of the weights being updated.

```
improv_rnn_train \
--config=attention_improv \
--run_dir=/tmp/improv_rnn/logdir/run1 \
--sequence_example_file=/tmp/improv_rnn/sequence_examples/eval_lead_sheets.tfrecord \
--hparams="{'batch_size':64,'rnn_layer_sizes':[64,64]}" \
--num_training_steps=20000 \
--eval
```

Run TensorBoard to view the training and evaluation data.

```
tensorboard --logdir=/tmp/improv_rnn/logdir
```

Then go to [http://localhost:6006](http://localhost:6006) to view the TensorBoard dashboard.

### Generate Melodies over Chords

Melodies can be generated during or after training. Run the command below to generate a set of melodies using the latest checkpoint file of your trained model.

`--run_dir` should be the same directory used for the training job. The `train` subdirectory within `--run_dir` is where the latest checkpoint file will be loaded from. For example, if we use `--run_dir=/tmp/improv_rnn/logdir/run1`. The most recent checkpoint file in `/tmp/improv_rnn/logdir/run1/train` will be used.

`--hparams` should be the same hyperparameters used for the training job, although some of them will be ignored, like the batch size.

`--output_dir` is where the generated MIDI files will be saved. `--num_outputs` is the number of melodies that will be generated. If `--render_chords` is specified, the chords over which the melody was generated will also be rendered to the MIDI file as notes.

At least one note needs to be fed to the model before it can start generating consecutive notes. We can use `--primer_melody` to specify a priming melody using a string representation of a Python list. The values in the list should be ints that follow the melodies_lib.Melody format (-2 = no event, -1 = note-off event, values 0 through 127 = note-on event for that MIDI pitch). For example `--primer_melody="[60, -2, 60, -2, 67, -2, 67, -2]"` would prime the model with the first four notes of Twinkle Twinkle Little Star. Instead of using `--primer_melody`, we can use `--primer_midi` to prime our model with a melody stored in a MIDI file.

In addition, the backing chord progression must be provided using `--backing_chords`, a string representation of the backing chords separated by spaces. For example, `--backing_chords="Am Dm G C F Bdim E E"` uses the chords from I Will Survive. By default, each chord will last 16 steps (a single measure), but `--steps_per_chord` can also be set to a different value.

```
improv_rnn_generate \
--config=attention_improv \
--run_dir=/tmp/improv_rnn/logdir/run1 \
--output_dir=/tmp/improv_rnn/generated \
--num_outputs=10 \
--primer_melody="[57]" \
--backing_chords="Am Dm G C F Bdim E E" \
--render_chords
```

### Creating a Bundle File

The [bundle format](/magenta/protobuf/generator.proto)
is a convenient way of combining the model checkpoint, metagraph, and
some metadata about the model into a single file.

To generate a bundle, use the
[create_bundle_file](/magenta/music/sequence_generator.py)
method within SequenceGenerator. Our generator script
supports a ```--save_generator_bundle``` flag that calls this method. Example:

```sh
improv_rnn_generate \
--config=attention_improv \
--run_dir=/tmp/improv_rnn/logdir/run1 \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--bundle_file=/tmp/improv_rnn.mag \
--save_generator_bundle
```
