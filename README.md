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

Next, install Bazel and TensorFlow. You can find instructions for the former [here](http://www.bazel.io/docs/install.html) and the latter [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md). After that's done, run the tests with this command:

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
