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
"""Build the generator for the lookback RNN model."""

# internal imports
from magenta.models.attention_rnn import attention_rnn_encoder_decoder
from magenta.models.attention_rnn import attention_rnn_graph
from magenta.models.shared import melody_rnn_sequence_generator
from magenta.protobuf import generator_pb2

DEFAULT_ID = 'attention_rnn'


def create_generator(checkpoint,
                     bundle,
                     steps_per_quarter=4,
                     hparams=None,
                     generator_id=DEFAULT_ID):
  melody_encoder_decoder = attention_rnn_encoder_decoder.MelodyEncoderDecoder()
  details = generator_pb2.GeneratorDetails(
      id=generator_id, description='Attention RNN Generator')
  return melody_rnn_sequence_generator.MelodyRnnSequenceGenerator(
      details,
      checkpoint,
      bundle,
      melody_encoder_decoder,
      attention_rnn_graph.build_graph,
      steps_per_quarter,
      {} if hparams is None else hparams)
