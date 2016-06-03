## About this project

Magenta encompasses two goals. It’s first a research project to advance the state-of-the art in music, video, image, and text generation. Second, it's an attempt to build a community of artists, coders, and machine learning researchers. To facilitate those goals, we are developing open-source infrastructure in this repository.

## Installation
The installation has three components. You are going to need Bazel to build packages, Tensorflow to run models, and an up to date version of this repository.

First, clone this repository:

```git clone https://github.com/tensorflow/magenta-staging.git```

Next, install Bazel and Tensorflow. You can find instructions for the former [here](http://www.bazel.io/docs/install.html) and the latter [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup). After that's done, run the tests with this command:

```bazel test magenta/...```

## Building your Dataset
Now that you have a working copy of Magenta, let's build your first MIDI dataset. We do this by creating a directory of MIDI files and converting them into NoteSequences. If you don't have any MIDIs handy, you can find some [here](http://www.midiworld.com/files/142/) from MidiWorld.

To run the script, first build it:

```bazel build magenta:convert_midi_dir_to_note_sequences```

Then run it:

```
./bazel-bin/magenta/convert_midi_dir_to_note_sequences \
--midi_dir=/path/to/midi/dir \
--output_file=/path/to/tfrecord/file \
--recursive
```
