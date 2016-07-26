# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate melodies from a trained checkpoint of a melody RNN model."""

import ast
import os
import random
import time

# internal imports
import tensorflow as tf

from magenta.lib import melodies_lib
from magenta.lib import sequence_generator
from magenta.lib import sequences_lib
from magenta.protobuf import generator_pb2

class MelodyRnnSequenceGenerator(sequence_generator.BaseSequenceGenerator):
  def __init__(self, details, checkpoint_file, melody_encoder_decoder,
               build_graph, hparams):
    """
    Args:
      details: A generator_pb2.GeneratorDetails for this generator.
      checkpoint_file: Where to search for the most recent model checkpoint.
      melody_encoder_decoder: A melodies_lib.MelodyEncoderDecoder object
          specific to your model.
      build_graph: A function that when called, returns the tf.Graph object for
          your model. The function will be passed the parameters:
          (mode, hparams_string, input_size, num_classes, sequence_example_file)
          For an example usage, see models/basic_rnn/basic_rnn_graph.py.
    """
    super(MelodyRnnSequenceGenerator, self).__init__(details, checkpoint_file)
    self._melody_encoder_decoder = melody_encoder_decoder
    self._build_graph = build_graph
    self._session = None

    self._hparams = hparams
    self._hparams['dropout_keep_prob'] = 1.0
    self._hparams['batch_size'] = 1

  def _initialize(self, checkpoint_file):
    graph = self._build_graph('generate',
                              repr(self._hparams),
                              self._melody_encoder_decoder.input_size,
                              self._melody_encoder_decoder.num_classes)
    with graph.as_default():
      saver = tf.train.Saver()
      self._session = tf.Session()
      checkpoint_file = tf.train.latest_checkpoint(checkpoint_file)
      tf.logging.info('Checkpoint used: %s', checkpoint_file)
      saver.restore(self._session, checkpoint_file)

  def _close(self):
    self._session.close()
    self._session = None

  def _generate(self, generate_sequence_request):
    melody = melodies_lib.MonophonicMelody()
    bpm = melodies_lib.DEFAULT_BEATS_PER_MINUTE

    primer_sequence = generate_sequence_request.input_sequence

    quantized_sequence = sequences_lib.QuantizedSequence()
    quantized_sequence.from_note_sequence(
        primer_sequence, melodies_lib.DEFAULT_STEPS_PER_BEAT)
    extracted_melodies = melodies_lib.extract_melodies(
        quantized_sequence, min_bars=0, min_unique_pitches=1,
        gap_bars=float('inf'), ignore_polyphonic_notes=True)
    if extracted_melodies and extracted_melodies[0].events:
      melody = extracted_melodies[0]
      bpm = quantized_sequence.bpm
    else:
      tf.logging.warn('No melodies were extracted from the priming sequence. '
                      'Melodies will be generated from scratch.')
      melody.events = [
          random.randint(self._melody_encoder_decoder.min_note,
                         self._melody_encoder_decoder.max_note)]

    transpose_amount = melody.squash(
        self._melody_encoder_decoder.min_note,
        self._melody_encoder_decoder.max_note,
        self._melody_encoder_decoder.transpose_to_key)

    inputs = self._session.graph.get_collection('inputs')[0]
    initial_state = self._session.graph.get_collection('initial_state')[0]
    final_state = self._session.graph.get_collection('final_state')[0]
    softmax = self._session.graph.get_collection('softmax')[0]

    # derive this from GenerateSequenceRequest
    num_steps = 128
    final_state_ = None
    for i in xrange(num_steps - len(melody)):
      if i == 0:
        inputs_ = self._melody_encoder_decoder.get_inputs_batch(
            [melody], full_length=True)
        initial_state_ = self._session.run(initial_state)
      else:
        inputs_ = self._melody_encoder_decoder.get_inputs_batch([melody])
        initial_state_ = final_state_

      feed_dict = {inputs: inputs_, initial_state: initial_state_}
      final_state_, softmax_ = self._session.run(
          [final_state, softmax], feed_dict)
      self._melody_encoder_decoder.extend_melodies([melody], softmax_)

    melody.transpose(-transpose_amount)

    generate_response = generator_pb2.GenerateSequenceResponse()
    generate_response.generated_sequence.CopyFrom(melody.to_sequence(bpm=bpm))
    return generate_response
