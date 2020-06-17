## Performance RNN

Performance RNN models polyphonic performances with dynamics and expressive timing. It uses an event sequence encoding like [Polyphony RNN](/magenta/models/polyphony_rnn/README.md) but with the following event types:

* NOTE_ON(*pitch*): start a note at *pitch*
* NOTE_OFF(*pitch*): stop a note at *pitch*
* TIME_SHIFT(*amount*): advance time by *amount*
* VELOCITY(*value*): change current velocity to *value*

This model creates music in a language similar to MIDI itself, with **note-on** and **note-off** events instead of explicit durations. In order to support expressive timing, the model controls the clock with **time-shift** events that move forward at increments of 10 ms, up to 1 second. All note-on and note-off events happen at the current time as determined by all previous time shifts in the event sequence. The model also supports **velocity** events that set the current velocity, used by subsequent note-on events.  Velocity can optionally be quantized into fewer than the 127 valid MIDI velocities.

Because of this representation, the model is capable of generating performances with more natural timing and dynamics compared to our other models that a) use a quantized metrical grid with fixed tempo and b) don't handle velocity.

At generation time, a few undesired behaviors can occur: note-off events with no previous note-on (these are ignored), and note-on events with no subsequent note-off (these are ended after 5 seconds).

## Web Interface

You can run Performance RNN live in your browser at the [Performance RNN browser demo](https://goo.gl/magenta/performancernn-demo), made with [TensorFlow.js](https://js.tensorflow.org). More details about the web port can be found at our blog post: [Real-time Performance RNN in the Browser](https://magenta.tensorflow.org/performance-rnn-browser).

## Colab and Jupyter notebooks

You can try out Performance RNN in the [Colab](https://colab.research.google.com) environment with the [performance_rnn.ipynb](https://colab.research.google.com/notebook#fileId=/v2/external/notebooks/magenta/performance_rnn/performance_rnn.ipynb) notebook.

There is also a Jupyter notebook [Performance_RNN.ipynb](https://github.com/tensorflow/magenta-demos/blob/master/jupyter-notebooks/Performance_RNN.ipynb)
in our [Magenta Demos](https://github.com/tensorflow/magenta-demos) repository showing how to generate performances from a trained model.

## How to Use

If you would like to run the model locally, first, set up your [Magenta environment](/README.md). Next, you can either use a pre-trained model or train your own.

## Pre-trained

If you want to get started right away, you can use a few models that we've pre-trained on [real performances from the Yamaha e-Piano Competition](http://www.piano-e-competition.com/midiinstructions.asp):

* [performance](http://download.magenta.tensorflow.org/models/performance.mag)
* [performance_with_dynamics](http://download.magenta.tensorflow.org/models/performance_with_dynamics.mag)
* [performance_with_dynamics_and_modulo_encoding](http://download.magenta.tensorflow.org/models/performance_with_dynamics_and_modulo_encoding.mag)
* [density_conditioned_performance_with_dynamics](http://download.magenta.tensorflow.org/models/density_conditioned_performance_with_dynamics.mag)
* [pitch_conditioned_performance_with_dynamics](http://download.magenta.tensorflow.org/models/pitch_conditioned_performance_with_dynamics.mag)
* [multiconditioned_performance_with_dynamics](http://download.magenta.tensorflow.org/models/multiconditioned_performance_with_dynamics.mag)

The bundle filenames correspond to the configs defined in [performance_model.py](/magenta/models/performance_rnn/performance_model.py). The first three models use different performance representations. The first, `performance`, ignores note velocities but models note on/off events with expressive timing. The `performance_with_dynamics` model includes velocity changes quantized into 32 bins. The `performance_with_dynamics_and_modulo_encoding` model uses an alternate encoding designed by [Vida Vakilotojar](https://github.com/vidavakil) where event values are mapped to points on the unit circle.

The latter three models are *conditional* models that can generate performances conditioned on desired note density, desired pitch class distribution, or both, respectively.

### Generate a performance

```
BUNDLE_PATH=<absolute path of .mag file>
CONFIG=<one of 'performance', 'performance_with_dynamics', etc., matching the bundle>

performance_rnn_generate \
--config=${CONFIG} \
--bundle_file=${BUNDLE_PATH} \
--output_dir=/tmp/performance_rnn/generated \
--num_outputs=10 \
--num_steps=3000 \
--primer_melody="[60,62,64,65,67,69,71,72]"
```

This will generate a performance starting with an ascending C major scale.

There are several command-line options for controlling the generation process:

* **primer_pitches**: A string representation of a Python list of pitches that will be used as a starting chord with a short duration. For example: ```"[60, 64, 67]"```.
* **primer_melody**: A string representation of a Python list of `note_seq.Melody` event values (-2 = no event, -1 = note-off event, values 0 through 127 = note-on event for that MIDI pitch). For example: `"[60, -2, 60, -2, 67, -2, 67, -2]"`.
* **primer_midi**: The path to a MIDI file containing a polyphonic track that will be used as a priming track.

If you're using one of the conditional models, there are additional command-line options you can use:

* **notes_per_second**: The desired number of notes per second in the output performance. Note that increasing this value will cause generation to take longer, as the number of RNN steps is roughly proportional to the number of notes generated.
* **pitch_class_histogram**: A string representation of a Python list of 12 values representing the relative frequency of notes of each pitch class, starting with C. For example: `"[2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]"` will tend to stick to a C-major scale, with twice as much C as any of the other notes of the scale.

These control variables are not strictly enforced, but can be used to guide the model's output. Currently these can only be set globally, affecting the entire performance.

For a full list of command line options, run `performance_rnn_generate --help`.

## Train your own

### Create NoteSequences

Our first step will be to convert a collection of MIDI or MusicXML files into NoteSequences. NoteSequences are [protocol buffers](https://developers.google.com/protocol-buffers/), which is a fast and efficient data format, and easier to work with than MIDI files. See [Building your Dataset](/magenta/scripts/README.md) for instructions on generating a TFRecord file of NoteSequences. In this example, we assume the NoteSequences were output to ```/tmp/notesequences.tfrecord```.

### Create SequenceExamples

SequenceExamples are fed into the model during training and evaluation. Each SequenceExample will contain a sequence of inputs and a sequence of labels that represent a performance. Run the command below to extract performances  from your NoteSequences and save them as SequenceExamples. Two collections of SequenceExamples will be generated, one for training, and one for evaluation, where the fraction of SequenceExamples in the evaluation set is determined by `--eval_ratio`. With an eval ratio of 0.10, 10% of the extracted performances will be saved in the eval collection, and 90% will be saved in the training collection.

If you are training an unconditioned model with note velocities, we recommend using the `performance_with_dynamics_compact` config, as the size of your TFRecord file will be *much* smaller.

```
CONFIG=<one of 'performance', 'performance_with_dynamics', etc.>

performance_rnn_create_dataset \
--config=${CONFIG} \
--input=/tmp/notesequences.tfrecord \
--output_dir=/tmp/performance_rnn/sequence_examples \
--eval_ratio=0.10
```

### Train and Evaluate the Model

Run the command below to start a training job. `--config` should match the configuration used when creating the dataset. `--run_dir` is the directory where checkpoints and TensorBoard data for this run will be stored. `--sequence_example_file` is the TFRecord file of SequenceExamples that will be fed to the model. `--num_training_steps` (optional) is how many update steps to take before exiting the training loop. If left unspecified, the training loop will run until terminated manually. `--hparams` (optional) can be used to specify hyperparameters other than the defaults.

```
performance_rnn_train \
--config=${CONFIG} \
--run_dir=/tmp/performance_rnn/logdir/run1 \
--sequence_example_file=/tmp/performance_rnn/sequence_examples/training_performances.tfrecord
```

Optionally run an eval job in parallel. `--run_dir`, `--hparams`, and `--num_training_steps` should all be the same values used for the training job. `--sequence_example_file` should point to the separate set of eval performances. Include `--eval` to make this an eval job, resulting in the model only being evaluated without any of the weights being updated.

```
performance_rnn_train \
--config=${CONFIG} \
--run_dir=/tmp/performance_rnn/logdir/run1 \
--sequence_example_file=/tmp/performance_rnn/sequence_examples/eval_performances.tfrecord \
--eval
```

Run TensorBoard to view the training and evaluation data.

```
tensorboard --logdir=/tmp/performance_rnn/logdir
```

Then go to [http://localhost:6006](http://localhost:6006) to view the TensorBoard dashboard.

### Generate Performances

Performances can be generated during or after training. Run the command below to generate a set of performances using the latest checkpoint file of your trained model.

`--run_dir` should be the same directory used for the training job. The `train` subdirectory within `--run_dir` is where the latest checkpoint file will be loaded from. For example, if we use `--run_dir=/tmp/performance_rnn/logdir/run1`. The most recent checkpoint file in `/tmp/performance_rnn/logdir/run1/train` will be used.

`--config` should be the same as used for the training job.

`--hparams` should be the same hyperparameters used for the training job, although some of them will be ignored, like the batch size.

`--output_dir` is where the generated MIDI files will be saved. `--num_outputs` is the number of performances that will be generated. `--num_steps` is how long each performance will be in steps, where one step is 10 ms (e.g. 3000 steps is 30 seconds).

See above for more information on other command line options.

```
performance_rnn_generate \
--run_dir=/tmp/performance_rnn/logdir/run1 \
--output_dir=/tmp/performance_rnn/generated \
--config=${CONFIG} \
--num_outputs=10 \
--num_steps=3000 \
--primer_melody="[60,62,64,65,67,69,71,72]"
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
performance_rnn_generate \
  --config=${CONFIG} \
  --run_dir=/tmp/performance_rnn/logdir/run1 \
  --bundle_file=/tmp/performance_rnn.mag \
  --save_generator_bundle
```
