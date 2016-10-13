# Models

This directory contains the various Magenta models.

## Generators
All generators for NoteSequence-based models should expose their generator
functionality by implementing the BaseSequenceGenerator abstract class defined
in
[music/sequence_generator.py](/magenta/music/sequence_generator.py).
This allows all generators to communicate with a standard protocol (defined in
[protobuf/genator.proto](/magenta/protobuf/generator.proto))
and will make it easier for various interfaces (e.g., MIDI controllers) to
communicate with any model.