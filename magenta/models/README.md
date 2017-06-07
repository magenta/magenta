# Models

This directory contains Magenta models.

## NSynth

[Neural Audio Synthesis]](/magenta/models/nsynth) as described in
[NSynth: Neural Audio Synthesis with WaveNet Autoencoders](https://arxiv.org/abs/1704.01279).


## Image stylization

This is the [Multistyle Pastiche Generator
model](/magenta/models/image_stylization) described in
[A Learned Representation For Artistic Style](https://arxiv.org/abs/1610.07629).
It generates artistic representations of photographs.

## Generators
All generators for NoteSequence-based models should expose their generator
functionality by implementing the BaseSequenceGenerator abstract class defined
in
[music/sequence_generator.py](/magenta/music/sequence_generator.py).
This allows all generators to communicate with a standard protocol (defined in
[protobuf/genator.proto](/magenta/protobuf/generator.proto))
and will make it easier for various interfaces (e.g., MIDI controllers) to
communicate with any model.
