## Pianoroll RNN-NADE

This model applies language modeling to polyphonic music generation using an
LSTM combined with a [NADE](https://arxiv.org/abs/1605.02226), an architecture
called an [RNN-NADE](http://www-etud.iro.umontreal.ca/~boulanni/ICML2012.pdf).
Unlike melody RNNs, this model needs to be capable of modeling multiple
simultaneous notes. It does so by representing a NoteSequence as a "pianoroll",
named after the medium used to store scores for
[player pianos](https://en.wikipedia.org/wiki/Piano_roll).

In a pianoroll, the score is represented as a binary matrix where each row
represents a step and each column represents a pitch. When a column has a value
of 1, that means the associated pitch is active at that time. When the column
value is 0, the pitch is inactivate. A downside of this representation is that
it is difficult to represent repeated legatto notes since they will appear as a
single note in a pianoroll.

Since we need to output multiple pitches at each step, we cannot use a softmax.
The [polyphony_rnn](/models/polyphony_rnn/README.md) posted previously skirted
this issue by representing a single time step as multiple, sequential outputs
from the RNN. In this model, we instead use a Neural Autoregressive Distribution
Estimator, or NADE to sample multiple outputs given the RNN state at each step.
See the original [RNN-NADE paper](http://www-etud.iro.umontreal.ca/~boulanni/ICML2012.pdf)
and our code for more details on how this architecture works.

## How to Use

First, set up your [Magenta environment](/README.md). Next, you can either use a pre-trained model or train your own.

## Pre-trained

If you want to get started right away, you can use a model that we've pre-trained:

* [pianoroll_rnn_nade](http://download.magenta.tensorflow.org/models/pianoroll_rnn_nade.mag): Trained
  on a large corpus of piano music scraped from the web.
* [pianoroll_rnn_nade-bach](http://download.magenta.tensorflow.org/models/pianoroll_rnn_nade-bach.mag):
  Trained on the [Bach Chorales](https://web.archive.org/web/20150503021418/http://www.jsbchorales.net/xml.shtml) dataset.

### Generate a pianoroll sequence

```
BUNDLE_PATH=<absolute path of .mag file>

pianoroll_rnn_nade_generate \
--bundle_file=${BUNDLE_PATH} \
--output_dir=/tmp/pianoroll_rnn_nade/generated \
--num_outputs=10 \
--num_steps=128 \
--primer_pitches="[67,64,60]"
```

This will generate a polyphonic pianoroll sequence using a C Major chord as a primer.

There are several command line options for controlling the generation process:

* **primer_pitches**: A string representation of a Python list of pitches that will be used as a starting chord with a quarter note duration. For example: ```"[60, 64, 67]"```.
* **primer_pianoroll**: A string representation of a Python list of `note_seq.PianorollSequence` event values (tuples of active MIDI pitches for a sequence of steps). For example: `"[(55,), (54,), (55, 53), (50,), (62, 52), (), (63, 55)]"`.
* **primer_midi**: The path to a MIDI file containing a polyphonic track that will be used as a priming track.

For a full list of command line options, run `pianoroll_rnn_nade_generate --help`.

Here's an example that is primed with two bars of
*Twinkle, Twinkle, Little Star* [set in two-voice counterpoint](http://www.noteflight.com/scores/view/2bd64f53ef4a4ec692f5be310780b634b2b5d98b):
```
BUNDLE_PATH=<absolute path of .mag file>

pianoroll_rnn_nade_generate \
--bundle_file=${BUNDLE_PATH} \
--output_dir=/tmp/pianoroll_rnn_nade/generated \
--qpm=90 \
--num_outputs=10 \
--num_steps=64 \
--primer_pianoroll="[(55,), (54,), (55, 52), (50,), (62, 52), (57,), "\
"(62, 55), (59,), (64, 52), (60,), (64, 59), (57,), (62, 59), (62, 55), "\
"(62, 52), (62, 55)]"
```

## Train your own

### Create NoteSequences

Our first step will be to convert a collection of MIDI or MusicXML files into NoteSequences. NoteSequences are [protocol buffers](https://developers.google.com/protocol-buffers/), which is a fast and efficient data format, and easier to work with than MIDI files. See [Building your Dataset](/magenta/scripts/README.md) for instructions on generating a TFRecord file of NoteSequences. In this example, we assume the NoteSequences were output to ```/tmp/notesequences.tfrecord```.

An example of training input is the Bach Chorales dataset, which will teach the model to generate sequences that sound like Bach. It is available either on this [archive.org mirror](https://web.archive.org/web/20150503021418/http://www.jsbchorales.net/xml.shtml) (the [original site](http://www.jsbchorales.net/xml.shtml) seems to be down) or via the [music21 bach corpus](https://github.com/cuthbertLab/music21/tree/master/music21/corpus/bach) (which also contains some additional Bach pieces).

### Create SequenceExamples

SequenceExamples are fed into the model during training and evaluation. Each SequenceExample will contain a sequence of inputs and a sequence of labels that represent a pianoroll sequence. Run the command below to extract pianoroll sequences from your NoteSequences and save them as SequenceExamples. Two collections of SequenceExamples will be generated, one for training, and one for evaluation, where the fraction of SequenceExamples in the evaluation set is determined by `--eval_ratio`. With an eval ratio of 0.10, 10% of the extracted polyphonic tracks will be saved in the eval collection, and 90% will be saved in the training collection.

```
pianoroll_rnn_nade_create_dataset \
--input=/tmp/notesequences.tfrecord \
--output_dir=/tmp/pianoroll_rnn_nade/sequence_examples \
--eval_ratio=0.10
```

### Train and Evaluate the Model

Run the command below to start a training job using the attention configuration. `--run_dir` is the directory where checkpoints and TensorBoard data for this run will be stored. `--sequence_example_file` is the TFRecord file of SequenceExamples that will be fed to the model. `--num_training_steps` (optional) is how many update steps to take before exiting the training loop. If left unspecified, the training loop will run until terminated manually. `--hparams` (optional) can be used to specify hyperparameters other than the defaults. For this example, we specify a custom batch size of 64 instead of the default batch size of 48. Using smaller batch sizes can help reduce memory usage, which can resolve potential out-of-memory issues when training larger models. We'll also use a single-layer RNN with 128 units, instead of the default of 3 layers of 128 units each. This will make our model train faster. However, if you have enough compute power, you can try using larger layer sizes for better results.

```
pianoroll_rnn_nade_train \
--run_dir=/tmp/pianoroll_rnn_nade/logdir/run1 \
--sequence_example_file=/tmp/pianoroll_rnn_nade/sequence_examples/training_pianoroll_tracks.tfrecord \
--hparams="batch_size=48,rnn_layer_sizes=[128]" \
--num_training_steps=20000
```

Optionally run an eval job in parallel. `--run_dir`, `--hparams`, and `--num_training_steps` should all be the same values used for the training job. `--sequence_example_file` should point to the separate set of eval pianoroll tracks. Include `--eval` to make this an eval job, resulting in the model only being evaluated without any of the weights being updated.

```
pianoroll_rnn_nade_train \
--run_dir=/tmp/pianoroll_rnn_nade/logdir/run1 \
--sequence_example_file=/tmp/pianoroll_rnn_nade/sequence_examples/eval_pianoroll_tracks.tfrecord \
--hparams="batch_size=48,rnn_layer_sizes=[128]" \
--num_training_steps=20000 \
--eval
```

Run TensorBoard to view the training and evaluation data.

```
tensorboard --logdir=/tmp/pianoroll_rnn_nade/logdir
```

Then go to [http://localhost:6006](http://localhost:6006) to view the TensorBoard dashboard.

### Generate Pianoroll Tracks

Pianoroll tracks can be generated during or after training. Run the command below to generate a set of pianoroll tracks using the latest checkpoint file of your trained model.

`--run_dir` should be the same directory used for the training job. The `train` subdirectory within `--run_dir` is where the latest checkpoint file will be loaded from. For example, if we use `--run_dir=/tmp/pianoroll_rnn_nade/logdir/run1`. The most recent checkpoint file in `/tmp/pianoroll_rnn_nade/logdir/run1/train` will be used.

`--hparams` should be the same hyperparameters used for the training job, although some of them will be ignored, like the batch size.

`--output_dir` is where the generated MIDI files will be saved. `--num_outputs` is the number of pianoroll tracks that will be generated. `--num_steps` is how long each melody will be in 16th steps (128 steps = 8 bars).

See above for more information on other command line options.

```
pianoroll_rnn_nade_generate \
--run_dir=/tmp/pianoroll_rnn_nade/logdir/run1 \
--output_dir=/tmp/pianoroll_rnn_nade/generated \
--num_outputs=10 \
--num_steps=128 \
--primer_pitches="[67,64,60]"
```

### Creating a Bundle File

The [bundle format](https://github.com/magenta/note-seq/blob/master/note_seq/protobuf/generator.proto)
is a convenient way of combining the model checkpoint, metagraph, and
some metadata about the model into a single file.

To generate a bundle, use the
[create_bundle_file](/magenta/models/shared/sequence_generator.py)
method within SequenceGenerator. Our generator script
supports a ```--save_generator_bundle``` flag that calls this method. Example:

```
pianoroll_rnn_nade_generate \
--run_dir=/tmp/pianoroll_rnn_nade/logdir/run1 \
--bundle_file=/tmp/pianoroll_rnn_nade.mag \
--save_generator_bundle
```
