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
The installation has three components. You are going to need Bazel to build packages, TensorFlow to run models, and an up-to-date version of this repository.

First, clone this repository:

```git clone https://github.com/tensorflow/magenta.git```

Next, install Bazel and TensorFlow. You'll need at least version 0.2.3 for Bazel and at least version 0.9 for TensorFlow. You can find instructions for the former [here](http://www.bazel.io/docs/install.html) and the latter [here](https://github.com/tensorflow/tensorflow/blob/v0.9.0rc0/tensorflow/g3doc/get_started/os_setup.md). After that's done, run the tests with this command:

```bazel test //magenta:all```

## Building your Dataset
Now that you have a working copy of Magenta, let's build your first MIDI dataset. We do this by creating a directory of MIDI files and converting them into NoteSequences. If you don't have any MIDIs handy, you can find some [here](http://www.midiworld.com/files/142/) from MidiWorld.

Build and run the script. Warnings may be printed by the MIDI parser if it encounteres a malformed MIDI file but these can be safely ignored. MIDI files that cannot be parsed will be skipped.

```
MIDI_DIRECTORY=<folder containing MIDI files. can have child folders.>

# TFRecord file that will contain NoteSequence protocol buffers.
SEQUENCES_TFRECORD=/tmp/notesequences.tfrecord

bazel run //magenta:convert_midi_dir_to_note_sequences -- \
--midi_dir=$MIDI_DIRECTORY \
--output_file=$SEQUENCES_TFRECORD \
--recursive
```

Note: To build and run in seperate commands, run

```
bazel build //magenta:convert_midi_dir_to_note_sequences

./bazel-bin/magenta/convert_midi_dir_to_note_sequences \
--midi_dir=$MIDI_DIRECTORY \
--output_file=$SEQUENCES_TFRECORD \
--recursive
```

## Generating MIDI

To create your own melodies with TensorFlow, train a model on the dataset you built above and then use it to generate new sequences. Select a model below for further instructions.

**[Basic RNN](magenta/models/basic_rnn)**: A simple recurrent neural network for predicting melodies.
