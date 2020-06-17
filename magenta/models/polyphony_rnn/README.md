## Polyphony RNN

This model applies language modeling to polyphonic music generation using an LSTM. Unlike melodies, this model needs to be capable of modeling multiple simultaneous notes. Taking inspiration from [BachBot](http://bachbot.com/) (described in [*Automatic Stylistic Composition of Bach Choralies with Deep LSTM*](https://ismir2017.smcnus.org/wp-content/uploads/2017/10/156_Paper.pdf)), we model polyphony as a single stream of note events with special START, STEP_END, and END symbols. Within a step, notes are sorted by pitch in descending order.

For example, using the default quantizing resolution of 4 steps per quarte note, a sequence containing only a C Major chord with a duration of one quarter note would look like this:

```
START
NEW_NOTE, 67
NEW_NOTE, 64
NEW_NOTE, 60
STEP_END
CONTINUED_NOTE, 67
CONTINUED_NOTE, 64
CONTINUED_NOTE, 60
STEP_END
CONTINUED_NOTE, 67
CONTINUED_NOTE, 64
CONTINUED_NOTE, 60
STEP_END
CONTINUED_NOTE, 67
CONTINUED_NOTE, 64
CONTINUED_NOTE, 60
STEP_END
END
```

## How to Use

First, set up your [Magenta environment](/README.md). Next, you can either use a pre-trained model or train your own.

## Pre-trained

If you want to get started right away, you can use a model that we've pre-trained on Bach chorales:

* [polyphony_rnn](http://download.magenta.tensorflow.org/models/polyphony_rnn.mag)

### Generate a polyphonic sequence

```
BUNDLE_PATH=<absolute path of .mag file>

polyphony_rnn_generate \
--bundle_file=${BUNDLE_PATH} \
--output_dir=/tmp/polyphony_rnn/generated \
--num_outputs=10 \
--num_steps=128 \
--primer_pitches="[67,64,60]" \
--condition_on_primer=true \
--inject_primer_during_generation=false
```

This will generate a polyphonic sequence using a C Major chord as a primer.

There are several command line options for controlling the generation process:

* **primer_pitches**: A string representation of a Python list of pitches that will be used as a starting chord with a quarter note duration. For example: ```"[60, 64, 67]"```.
* **primer_melody**: A string representation of a Python list of `note_seq.Melody` event values (-2 = no event, -1 = note-off event, values 0 through 127 = note-on event for that MIDI pitch). For example: `"[60, -2, 60, -2, 67, -2, 67, -2]"`.
* **primer_midi**: The path to a MIDI file containing a polyphonic track that will be used as a priming track.
* **condition_on_primer**: If set, the RNN will receive the primer as its input before it begins generating a new sequence. You most likely want this to be true if you're using **primer_pitches** to start the sequence with a chord to establish a certain key. If you're using **primer_melody** because you want to inject a melody into the output using **inject_primer_during_generation**, you likely want this to be false, otherwise the model will see a monophonic melody before being asked to produce a polyphonic sequence. However, it may be interesting to experiment with this being on or off for each of those cases.
* **inject_primer_during_generation**: If set, the primer will be injected as a part of the generated sequence. This option is useful if you want the model to harmonize an existing melody. This option will most likely be used with **primer_melody** and `--condition_on_primer=false`.

For a full list of command line options, run `polyphony_rnn_generate --help`.

Here's another example that will harmonize the first few notes of *Twinkle, Twinkle, Little Star*:

```
BUNDLE_PATH=<absolute path of .mag file>

polyphony_rnn_generate \
--bundle_file=${BUNDLE_PATH} \
--output_dir=/tmp/polyphony_rnn/generated \
--num_outputs=10 \
--num_steps=64 \
--primer_melody="[60, -2, -2, -2, 60, -2, -2, -2, "\
"67, -2, -2, -2, 67, -2, -2, -2, 69, -2, -2, -2, "\
"69, -2, -2, -2, 67, -2, -2, -2, -2, -2, -2, -2]" \
--condition_on_primer=false \
--inject_primer_during_generation=true
```

Note that we set `--inject_primer_during_generation=true` so that the primer melody is injected to the event sequence during generation. We also set `--condition_on_primer=false` because it is unlikely that the model encountered a monophonic melody while training on the Bach chorales, so it may not make sense to condition on it.

## Train your own

### Create NoteSequences

Our first step will be to convert a collection of MIDI or MusicXML files into NoteSequences. NoteSequences are [protocol buffers](https://developers.google.com/protocol-buffers/), which is a fast and efficient data format, and easier to work with than MIDI files. See [Building your Dataset](/magenta/scripts/README.md) for instructions on generating a TFRecord file of NoteSequences. In this example, we assume the NoteSequences were output to ```/tmp/notesequences.tfrecord```.

If you want to build a model that is similar to [BachBot](http://bachbot.com), you can try training on the Bach Chorales dataset, which is available either on this [archive.org mirror](https://web.archive.org/web/20150503021418/http://www.jsbchorales.net/xml.shtml) (the [original site](http://www.jsbchorales.net/xml.shtml) seems to be down) or via the [music21 bach corpus](https://github.com/cuthbertLab/music21/tree/master/music21/corpus/bach) (which also contains some additional Bach pieces).

### Create SequenceExamples

SequenceExamples are fed into the model during training and evaluation. Each SequenceExample will contain a sequence of inputs and a sequence of labels that represent a polyphonic sequence. Run the command below to extract polyphonic sequences from your NoteSequences and save them as SequenceExamples. Two collections of SequenceExamples will be generated, one for training, and one for evaluation, where the fraction of SequenceExamples in the evaluation set is determined by `--eval_ratio`. With an eval ratio of 0.10, 10% of the extracted polyphonic tracks will be saved in the eval collection, and 90% will be saved in the training collection.

```
polyphony_rnn_create_dataset \
--input=/tmp/notesequences.tfrecord \
--output_dir=/tmp/polyphony_rnn/sequence_examples \
--eval_ratio=0.10
```

### Train and Evaluate the Model

Run the command below to start a training job using the attention configuration. `--run_dir` is the directory where checkpoints and TensorBoard data for this run will be stored. `--sequence_example_file` is the TFRecord file of SequenceExamples that will be fed to the model. `--num_training_steps` (optional) is how many update steps to take before exiting the training loop. If left unspecified, the training loop will run until terminated manually. `--hparams` (optional) can be used to specify hyperparameters other than the defaults. For this example, we specify a custom batch size of 64 instead of the default batch size of 128. Using smaller batch sizes can help reduce memory usage, which can resolve potential out-of-memory issues when training larger models. We'll also use a 2-layer RNN with 64 units each, instead of the default of 3 layers of 256 units each. This will make our model train faster. However, if you have enough compute power, you can try using larger layer sizes for better results.

```
polyphony_rnn_train \
--run_dir=/tmp/polyphony_rnn/logdir/run1 \
--sequence_example_file=/tmp/polyphony_rnn/sequence_examples/training_poly_tracks.tfrecord \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--num_training_steps=20000
```

Optionally run an eval job in parallel. `--run_dir`, `--hparams`, and `--num_training_steps` should all be the same values used for the training job. `--sequence_example_file` should point to the separate set of eval polyphonic tracks. Include `--eval` to make this an eval job, resulting in the model only being evaluated without any of the weights being updated.

```
polyphony_rnn_train \
--run_dir=/tmp/polyphony_rnn/logdir/run1 \
--sequence_example_file=/tmp/polyphony_rnn/sequence_examples/eval_poly_tracks.tfrecord \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--num_eval_examples=20000 \
--eval
```

Run TensorBoard to view the training and evaluation data.

```
tensorboard --logdir=/tmp/polyphony_rnn/logdir
```

Then go to [http://localhost:6006](http://localhost:6006) to view the TensorBoard dashboard.

### Generate Polyphonic Tracks

Polyphonic tracks can be generated during or after training. Run the command below to generate a set of polyphonic tracks using the latest checkpoint file of your trained model.

`--run_dir` should be the same directory used for the training job. The `train` subdirectory within `--run_dir` is where the latest checkpoint file will be loaded from. For example, if we use `--run_dir=/tmp/polyphony_rnn/logdir/run1`. The most recent checkpoint file in `/tmp/polyphony_rnn/logdir/run1/train` will be used.

`--hparams` should be the same hyperparameters used for the training job, although some of them will be ignored, like the batch size.

`--output_dir` is where the generated MIDI files will be saved. `--num_outputs` is the number of polyphonic tracks that will be generated. `--num_steps` is how long each melody will be in 16th steps (128 steps = 8 bars).

See above for more information on other command line options.

```
polyphony_rnn_generate \
--run_dir=/tmp/polyphony_rnn/logdir/run1 \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--output_dir=/tmp/polyphony_rnn/generated \
--num_outputs=10 \
--num_steps=128 \
--primer_pitches="[67,64,60]" \
--condition_on_primer=true \
--inject_primer_during_generation=false
```

### Creating a Bundle File

The [bundle format](https://github.com/magenta/note-seq/blob/master/note_seq/protobuf/generator.proto)
is a convenient way of combining the model checkpoint, metagraph, and
some metadata about the model into a single file.

To generate a bundle, use the [create_bundle_file](/magenta/models/shared/sequence_generator.py) method within SequenceGenerator. Our generator script supports a `--save_generator_bundle` flag that calls this method. When using the `--save_generator_bundle` mode, you need to supply the `--hparams` flag with the same values used during training.

Example:

```
polyphony_rnn_generate \
--run_dir=/tmp/polyphony_rnn/logdir/run1 \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--bundle_file=/tmp/polyphony_rnn.mag \
--save_generator_bundle
```
