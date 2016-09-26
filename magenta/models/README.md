# Models

This directory contains the various Magenta models.

## Generators
All generators for NoteSequence-based models should expose their generator
functionality by implementing the BaseSequenceGenerator abstract class defined
in
[lib/sequence_generator.py](https://github.com/tensorflow/magenta/blob/master/magenta/music/sequence_generator.py).
This allows all generators to communicate with a standard protocol (defined in
[protobuf/genator.proto](https://github.com/tensorflow/magenta/blob/master/magenta/protobuf/generator.proto))
and will make it easier for various interfaces (e.g., MIDI controllers) to
communicate with any model.

Note that the Melody models share a common implementation of the
BaseSequenceGenerator interface in
[MelodyRnnSequenceGenerator](https://github.com/tensorflow/magenta/blob/master/magenta/models/shared/melody_rnn_sequence_generator.py),
which they then invoke using the generate scripts in each model's directory
(e.g.,
[attention_rnn_generate.py](https://github.com/tensorflow/magenta/blob/master/magenta/models/attention_rnn/attention_rnn_generate.py)).
