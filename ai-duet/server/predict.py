#
# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import tempfile

import tensorflow as tf
import magenta
from magenta.protobuf import generator_pb2

steps_per_quarter = 4
qpm = 120
BACKING_CHORDS = "C G Am F C G Am F"


def improv_generator(bundle='chord_pitches_improv'):
    # keep imports inside since we only need one set at a time
    from magenta.models.improv_rnn import improv_rnn_model
    from magenta.models.improv_rnn import improv_rnn_sequence_generator
    from magenta.models.improv_rnn.improv_rnn_generate import CHORD_SYMBOL

    config = magenta.models.improv_rnn.improv_rnn_model.default_configs[bundle]
    bundle_file = magenta.music.read_bundle_file(os.path.abspath(bundle+'.mag'))

    generator = improv_rnn_sequence_generator.ImprovRnnSequenceGenerator(
      model=improv_rnn_model.ImprovRnnModel(config),
      details=config.details,
      steps_per_quarter=steps_per_quarter,
      bundle=bundle_file)

    def generate_backing_chords(input_sequence, generator_options):
        # Create backing chord progression from flags.
        steps_per_chord = steps_per_quarter
        backing_chords = BACKING_CHORDS
        raw_chords = backing_chords.split()
        repeated_chords = [chord for chord in raw_chords
                           for _ in range(steps_per_chord)]
        backing_chords = magenta.music.ChordProgression(repeated_chords)

        # Derive the total number of seconds to generate based on the QPM of the
        # priming sequence and the length of the backing chord progression.
        seconds_per_step = 60.0 / qpm / generator.steps_per_quarter
        total_seconds = len(backing_chords) * seconds_per_step

        # Specify start/stop time for generation based on starting generation at
        # the end of the priming sequence and continuing until the sequence is
        # num_steps long.
        last_end_time = (max(n.end_time for n in input_sequence.notes)
                         if input_sequence.notes else 0)
        generate_section = generator_options.generate_sections.add(
            start_time=last_end_time + seconds_per_step,
            end_time=total_seconds)

        if generate_section.start_time >= generate_section.end_time:
            tf.logging.fatal(
                'Priming sequence is longer than the total number of steps '
                'requested: Priming sequence length: %s, Generation length '
                'requested: %s',
                generate_section.start_time, total_seconds)
            return

        # Add the backing chords to the input sequence.
        chord_sequence = backing_chords.to_sequence(sequence_start_time=0.0,
                                                    qpm=qpm)
        for text_annotation in chord_sequence.text_annotations:
            if text_annotation.annotation_type == CHORD_SYMBOL:
                chord = input_sequence.text_annotations.add()
                chord.CopyFrom(text_annotation)
        input_sequence.total_time = len(backing_chords) * seconds_per_step
        return input_sequence

    old_generate = generator.generate

    def wrapped_generate(primer_sequence, generator_options):
        # throw out old options as they have section generation info
        new_generator_options = generator_pb2.GeneratorOptions()
        input_sequence = generate_backing_chords(primer_sequence,
                                                 new_generator_options)
        return old_generate(input_sequence, generator_options)

    generator.generate = wrapped_generate
    return generator


def drums_generator(bundle='drum_kit'):
    from magenta.models.drums_rnn import drums_rnn_model
    from magenta.models.drums_rnn import drums_rnn_sequence_generator

    config = magenta.models.drums_rnn.drums_rnn_model.default_configs[bundle]
    bundle_file = magenta.music.read_bundle_file(os.path.abspath(bundle+'.mag'))

    return drums_rnn_sequence_generator.DrumsRnnSequenceGenerator(
          model=drums_rnn_model.DrumsRnnModel(config),
          details=config.details,
          steps_per_quarter=steps_per_quarter,
          bundle=bundle_file)


def polyphony_generator(bundle='polyphony'):
    from magenta.models.polyphony_rnn import polyphony_model
    from magenta.models.polyphony_rnn import polyphony_sequence_generator

    config = magenta.models.polyphony_rnn.polyphony_model.default_configs[bundle]
    bundle_file = magenta.music.read_bundle_file(os.path.abspath(bundle+'.mag'))
    steps_per_quarter = 8

    return polyphony_sequence_generator.PolyphonyRnnSequenceGenerator(
          model=polyphony_model.PolyphonyRnnModel(config),
          details=config.details,
          steps_per_quarter=steps_per_quarter,
          bundle=bundle_file)


def melody_generator(bundle='attention_rnn'):
    from magenta.models.melody_rnn import melody_rnn_model
    from magenta.models.melody_rnn import melody_rnn_sequence_generator

    config = magenta.models.melody_rnn.melody_rnn_model.default_configs[bundle]
    bundle_file = magenta.music.read_bundle_file(os.path.abspath(bundle+'.mag'))
    steps_per_quarter = 4

    return melody_rnn_sequence_generator.MelodyRnnSequenceGenerator(
          model=melody_rnn_model.MelodyRnnModel(config),
          details=config.details,
          steps_per_quarter=steps_per_quarter,
          bundle=bundle_file)


generator = melody_generator()
# generator = improv_generator()
# generator = polyphony_generator()
# generator = drums_generator()


def _steps_to_seconds(steps, qpm):
    return steps * 60.0 / qpm / steps_per_quarter


def generate_midi(midi_data, total_seconds=10):
    primer_sequence = magenta.music.midi_io.midi_to_sequence_proto(midi_data)

    # predict the tempo
    if len(primer_sequence.notes) > 4:
        estimated_tempo = midi_data.estimate_tempo()
        if estimated_tempo > 240:
            qpm = estimated_tempo / 2
        else:
            qpm = estimated_tempo
    else:
        qpm = 120
    primer_sequence.tempos[0].qpm = qpm

    generator_options = generator_pb2.GeneratorOptions()
    # Set the start time to begin on the next step after the last note ends.
    last_end_time = (max(n.end_time for n in primer_sequence.notes)
                     if primer_sequence.notes else 0)
    generator_options.generate_sections.add(
        start_time=last_end_time + _steps_to_seconds(1, qpm),
        end_time=total_seconds)

    # generate the output sequence
    generated_sequence = generator.generate(primer_sequence, generator_options)

    output = tempfile.NamedTemporaryFile()
    magenta.music.midi_io.sequence_proto_to_midi_file(generated_sequence,
                                                      output.name)
    output.seek(0)
    return output
