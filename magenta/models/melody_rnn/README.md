## Melody RNN

This model applies language modeling to melody generation using an LSTM.

## Configurations

### Basic

This configuration acts as a baseline for melody generation with an LSTM model. It uses basic one-hot encoding to represent extracted melodies as input to the LSTM. For training, all examples are transposed to the MIDI pitch range \[48, 84\] and outputs will also be in this range.

### Mono

This configuration acts as a baseline for melody generation with an LSTM model. It uses basic one-hot encoding to represent extracted melodies as input to the LSTM. While `basic_rnn` is trained by transposing all inputs to a narrow range, `mono_rnn` is able to use the full 128 MIDI pitches.

### Lookback

Lookback RNN introduces custom inputs and labels. The custom inputs allow the model to more easily recognize patterns that occur across 1 and 2 bars. They also help the model recognize patterns related to an events position within the measure. The custom labels reduce the amount of information that the RNN’s cell state has to remember by allowing the model to more easily repeat events from 1 and 2 bars ago. This results in melodies that wander less and have a more musical structure. For more information about the custom inputs and labels, and to hear some generated sample melodies, check out the [blog post](https://magenta.tensorflow.org/2016/07/15/lookback-rnn-attention-rnn/). You can also read through the `events_to_input` and `events_to_label` methods in `note_seq/encoder_decoder.py` and `note_seq/melody_encoder_decoder.py` to see how the custom inputs and labels are actually being encoded.

### Attention

In this configuration we introduce the use of attention. Attention allows the model to more easily access past information without having to store that information in the RNN cell's state. This allows the model to more easily learn longer term dependencies, and results in melodies that have longer arching themes. For an overview of how the attention mechanism works and to hear some generated sample melodies, check out the [blog post](https://magenta.tensorflow.org/2016/07/15/lookback-rnn-attention-rnn/). You can also read through the `AttentionCellWrapper` code in Tensorflow to see what's really going on under the hood.

## How to Use

First, set up your [Magenta environment](/README.md). Next, you can either use a pre-trained model or train your own.

## Pre-trained

If you want to get started right away, you can use a model that we've pre-trained on thousands of MIDI files.
We host .mag bundle files for each of the configurations described above at these links:

* [basic_rnn](http://download.magenta.tensorflow.org/models/basic_rnn.mag)
* [mono_rnn](http://download.magenta.tensorflow.org/models/mono_rnn.mag)
* [lookback_rnn](http://download.magenta.tensorflow.org/models/lookback_rnn.mag)
* [attention_rnn](http://download.magenta.tensorflow.org/models/attention_rnn.mag)

### Generate a melody

```
BUNDLE_PATH=<absolute path of .mag file>
CONFIG=<one of 'basic_rnn', 'lookback_rnn', or 'attention_rnn', matching the bundle>

melody_rnn_generate \
--config=${CONFIG} \
--bundle_file=${BUNDLE_PATH} \
--output_dir=/tmp/melody_rnn/generated \
--num_outputs=10 \
--num_steps=128 \
--primer_melody="[60]"
```

This will generate a melody starting with a middle C. If you'd like, you can also supply a priming melody using a string representation of a Python list. The values in the list should be ints that follow the melodies_lib.Melody format (-2 = no event, -1 = note-off event, values 0 through 127 = note-on event for that MIDI pitch). For example `--primer_melody="[60, -2, 60, -2, 67, -2, 67, -2]"` would prime the model with the first four notes of *Twinkle Twinkle Little Star*. Instead of using `--primer_melody`, we can use `--primer_midi` to prime our model with a melody stored in a MIDI file. For example, `--primer_midi=<absolute path to magenta/models/melody_rnn/primer.mid>` will prime the model with the melody in that MIDI file.

## Train your own

### Create NoteSequences

Our first step will be to convert a collection of MIDI files into NoteSequences. NoteSequences are [protocol buffers](https://developers.google.com/protocol-buffers/), which is a fast and efficient data format, and easier to work with than MIDI files. See [Building your Dataset](/magenta/scripts/README.md) for instructions on generating a TFRecord file of NoteSequences. In this example, we assume the NoteSequences were output to ```/tmp/notesequences.tfrecord```.

### Create SequenceExamples

SequenceExamples are fed into the model during training and evaluation. Each SequenceExample will contain a sequence of inputs and a sequence of labels that represent a melody. Run the command below to extract melodies from our NoteSequences and save them as SequenceExamples. Two collections of SequenceExamples will be generated, one for training, and one for evaluation, where the fraction of SequenceExamples in the evaluation set is determined by `--eval_ratio`. With an eval ratio of 0.10, 10% of the extracted melodies will be saved in the eval collection, and 90% will be saved in the training collection.

```
melody_rnn_create_dataset \
--config=<one of 'basic_rnn', 'mono_rnn', lookback_rnn', or 'attention_rnn'> \
--input=/tmp/notesequences.tfrecord \
--output_dir=/tmp/melody_rnn/sequence_examples \
--eval_ratio=0.10
```

### Train and Evaluate the Model

Run the command below to start a training job using the attention configuration. `--run_dir` is the directory where checkpoints and TensorBoard data for this run will be stored. `--sequence_example_file` is the TFRecord file of SequenceExamples that will be fed to the model. `--num_training_steps` (optional) is how many update steps to take before exiting the training loop. If left unspecified, the training loop will run until terminated manually. `--hparams` (optional) can be used to specify hyperparameters other than the defaults. For this example, we specify a custom batch size of 64 instead of the default batch size of 128. Using smaller batch sizes can help reduce memory usage, which can resolve potential out-of-memory issues when training larger models. We'll also use a 2-layer RNN with 64 units each, instead of the default of 2 layers of 128 units each. This will make our model train faster. However, if you have enough compute power, you can try using larger layer sizes for better results. You can also adjust how many previous steps the attention mechanism looks at by changing the `attn_length` hyperparameter. For this example we leave it at the default value of 40 steps (2.5 bars).

```
melody_rnn_train \
--config=attention_rnn \
--run_dir=/tmp/melody_rnn/logdir/run1 \
--sequence_example_file=/tmp/melody_rnn/sequence_examples/training_melodies.tfrecord \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--num_training_steps=20000
```

Optionally run an eval job in parallel. `--run_dir`, `--hparams`, and `--num_training_steps` should all be the same values used for the training job. `--sequence_example_file` should point to the separate set of eval melodies. Include `--eval` to make this an eval job, resulting in the model only being evaluated without any of the weights being updated.

```
melody_rnn_train \
--config=attention_rnn \
--run_dir=/tmp/melody_rnn/logdir/run1 \
--sequence_example_file=/tmp/melody_rnn/sequence_examples/eval_melodies.tfrecord \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--num_training_steps=20000 \
--eval
```

Run TensorBoard to view the training and evaluation data.

```
tensorboard --logdir=/tmp/melody_rnn/logdir
```

Then go to [http://localhost:6006](http://localhost:6006) to view the TensorBoard dashboard.

### Generate Melodies

Melodies can be generated during or after training. Run the command below to generate a set of melodies using the latest checkpoint file of your trained model.

`--run_dir` should be the same directory used for the training job. The `train` subdirectory within `--run_dir` is where the latest checkpoint file will be loaded from. For example, if we use `--run_dir=/tmp/melody_rnn/logdir/run1`. The most recent checkpoint file in `/tmp/melody_rnn/logdir/run1/train` will be used.

`--hparams` should be the same hyperparameters used for the training job, although some of them will be ignored, like the batch size.

`--output_dir` is where the generated MIDI files will be saved. `--num_outputs` is the number of melodies that will be generated. `--num_steps` is how long each melody will be in 16th steps (128 steps = 8 bars).

At least one note needs to be fed to the model before it can start generating consecutive notes. We can use `--primer_melody` to specify a priming melody using a string representation of a Python list. The values in the list should be ints that follow the melodies_lib.Melody format (-2 = no event, -1 = note-off event, values 0 through 127 = note-on event for that MIDI pitch). For example `--primer_melody="[60, -2, 60, -2, 67, -2, 67, -2]"` would prime the model with the first four notes of Twinkle Twinkle Little Star. Instead of using `--primer_melody`, we can use `--primer_midi` to prime our model with a melody stored in a MIDI file. For example, `--primer_midi=<absolute path to magenta/models/shared/primer.mid>` will prime the model with the melody in that MIDI file. If neither `--primer_melody` nor `--primer_midi` are specified, a random note from the model's note range will be chosen as the first note, then the remaining notes will be generated by the model. In the example below we prime the melody with `--primer_melody="[60]"`, a single note-on event for the note C4.


```
melody_rnn_generate \
--config=attention_rnn \
--run_dir=/tmp/melody_rnn/logdir/run1 \
--output_dir=/tmp/melody_rnn/generated \
--num_outputs=10 \
--num_steps=128 \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--primer_melody="[60]"
```

### Creating a Bundle File

The [bundle format](https://github.com/magenta/note-seq/blob/master/note_seq/generator.proto)
is a convenient way of combining the model checkpoint, metagraph, and
some metadata about the model into a single file.

To generate a bundle, use the
[create_bundle_file](/magenta/models/shared/sequence_generator.py)
method within SequenceGenerator. All of our melody model generator scripts
support a ```--save_generator_bundle``` flag that calls this method. Example:

```sh
melody_rnn_generate \
--config=attention_rnn \
--run_dir=/tmp/melody_rnn/logdir/run1 \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--bundle_file=/tmp/attention_rnn.mag \
--save_generator_bundle
```
