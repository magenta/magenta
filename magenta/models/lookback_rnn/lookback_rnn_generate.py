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
r"""Generate melodies from a trained checkpoint of the lookback RNN model.

Example usage:
  $ bazel build magenta/models/lookback_rnn:lookback_rnn_generate

  $ ./bazel-bin/magenta/models/lookback_rnn/lookback_rnn_generate \
    --run_dir=/tmp/lookback_rnn/logdir/run1 \
    --output_dir=/tmp/lookback_rnn/generated \
    --num_outputs=10 \
    --num_steps=128 \
    --primer_melody="[60]"

See /magenta/models/shared/melody_rnn_generate.py for flag descriptions.
"""

# internal imports
import lookback_rnn_encoder_decoder
import lookback_rnn_graph
import tensorflow as tf

from magenta.models.shared import melody_rnn_generate


def main(unused_argv):
  melody_encoder_decoder = lookback_rnn_encoder_decoder.MelodyEncoderDecoder()
  melody_rnn_generate.run(melody_encoder_decoder,
                          lookback_rnn_graph.build_graph)


if __name__ == '__main__':
  tf.app.run()
