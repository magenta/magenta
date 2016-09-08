## Building your Dataset

After setting up your [Magenta environment](https://github.com/tensorflow/magenta/blob/master/README.md), you can build your first MIDI dataset. We do this by creating a directory of MIDI files and converting them into NoteSequences. If you don't have any MIDIs handy, you can use the [Lakh MIDI Dataset](http://colinraffel.com/projects/lmd/) or find some [at MidiWorld](http://www.midiworld.com/files/142/).

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

