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
"""Polyphonic RNN model."""

# internal imports

import magenta
from magenta.models.shared import events_rnn_model
from magenta.models.polyphonic_rnn import polyphony_encoder_decoder

default_configs = {
    'polyphony': events_rnn_model.EventSequenceRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='polyphony',
            description='Polyphonic RNN'),
        magenta.music.OneHotEventSequenceEncoderDecoder(
            polyphony_encoder_decoder.PolyphonyOneHotEncoding()),
        magenta.common.HParams(
            batch_size=128,
            rnn_layer_sizes=[128, 128],
            dropout_keep_prob=0.5,
            skip_first_n_losses=10,
            clip_norm=5,
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.95)),
}

