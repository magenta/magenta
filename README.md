<img src="http://magenta.tensorflow.org/assets/magenta-logo.png" height="75">

**Magenta** is a project from the [Google Brain team](https://research.google.com/teams/brain/)
that asks: Can we use machine learning to create compelling art and music? If
so, how? If not, why not?  We’ll use [TensorFlow](https://www.tensorflow.org),
and we’ll release our models and tools in open source on this GitHub. We’ll also
post demos, tutorial blog postings, and technical papers. Soon we’ll begin
accepting code contributions from the community at large. If you’d like to keep
up on Magenta as it grows, you can read our [blog](http://magenta.tensorflow.org) and or join our
[discussion group](http://groups.google.com/a/tensorflow.org/forum/#!forum/magenta-discuss).

## Installation

### Docker
The easiest way to get started with Magenta is to use our Docker container.
First, [install Docker](https://docs.docker.com/engine/installation/). Next, run
this command:

```docker run -it -v /tmp/magenta:/magenta-data tensorflow/magenta```

This will start a shell in a directory with all Magenta components compiled and
ready to run.

This also maps the directory ```/tmp/magenta``` on the host machine to
```/magenta-data``` within the Docker session. **WARNING**: only data saved in
```/magenta-data``` will persist across sessions.

One downside to the Docker container is that it is isolated from the host. If
you want to listen to a generated MIDI file, you'll need to copy it to the host
machine. Similarly, because our
[MIDI instrument interface](magenta/interfaces/midi) requires access to the host
MIDI port, it will not work within the Docker container. You'll need to use the
full Development Environment.

Note: Our docker image is also available at ```gcr.io/tensorflow/magenta```.

### Development Environment
If you want to develop on Magenta, use our
[MIDI instrument interface](magenta/interfaces/midi) or preview MIDI files
without copying them out out of the Docker environment, you'll need to set up
the full Development Environment.

The installation has three components. You are going to need Bazel to build packages, TensorFlow to run models, and an up-to-date version of this repository.

First, clone this repository:

```git clone https://github.com/tensorflow/magenta.git```

Next, [install Bazel](http://www.bazel.io/docs/install.html). We recommend the
latest version, currently 0.3.1.

Finally, [install TensorFlow](http://www.bazel.io/docs/install.html). We require
version 0.10 or later.

After that's done, run the tests with this command:

```bazel test //magenta/...```

## Building your Dataset
Now that you have a working copy of Magenta, let's build your first MIDI dataset. We do this by creating a directory of MIDI files and converting them into NoteSequences. If you don't have any MIDIs handy, you can find some [here](http://www.midiworld.com/files/142/) from MidiWorld.

Build and run the script. Warnings may be printed by the MIDI parser if it encounters a malformed MIDI file but these can be safely ignored. MIDI files that cannot be parsed will be skipped.

```
MIDI_DIRECTORY=<folder containing MIDI files. can have child folders.>

# TFRecord file that will contain NoteSequence protocol buffers.
SEQUENCES_TFRECORD=/tmp/notesequences.tfrecord

bazel run //magenta/scripts:convert_midi_dir_to_note_sequences -- \
--midi_dir=$MIDI_DIRECTORY \
--output_file=$SEQUENCES_TFRECORD \
--recursive
```

Note: To build and run in separate commands, run

```
bazel build //magenta/scripts:convert_midi_dir_to_note_sequences

./bazel-bin/magenta/scripts/convert_midi_dir_to_note_sequences \
--midi_dir=$MIDI_DIRECTORY \
--output_file=$SEQUENCES_TFRECORD \
--recursive
```

___Data processing APIs___

If you are interested in adding your own model, please take a look at how we create our datasets under the hood: [Data processing in Magenta](https://github.com/tensorflow/magenta/blob/master/magenta/pipelines)

## Generating MIDI

To create your own melodies with TensorFlow, train a model on the dataset you built above and then use it to generate new sequences. Select a model below for further instructions.

**[Basic RNN](magenta/models/basic_rnn)**: A simple recurrent neural network for predicting melodies.

**[Lookback RNN](magenta/models/lookback_rnn)**: A recurrent neural network for predicting melodies that uses custom inputs and labels.

**[Attention RNN](magenta/models/attention_rnn)**: A recurrent neural network for predicting melodies that uses attention.

## Using a MIDI Instrument

After you've trained one of the models above, you can use our [MIDI interface](magenta/interfaces/midi) to play with it interactively.
