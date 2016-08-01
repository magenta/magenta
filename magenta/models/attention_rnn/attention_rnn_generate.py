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
r"""Generate melodies from a trained checkpoint of the attention RNN model.

Example usage:
  $ bazel build magenta/models/attention_rnn:attention_rnn_generate

  $ ./bazel-bin/magenta/models/attention_rnn/attention_rnn_generate \
    --run_dir=/tmp/lookback_rnn/logdir/run1 \
    --output_dir=/tmp/lookback_rnn/generated \
    --num_outputs=10 \
    --num_steps=128 \
    --primer_melody="[60]"

See /magenta/models/shared/melody_rnn_generate.py for flag descriptions.
"""

# internal imports
import attention_rnn_generator
import tensorflow as tf

from magenta.models.shared import melody_rnn_generate


def main(unused_argv):
  with attention_rnn_generator.create_generator(
      melody_rnn_generate.get_train_dir(),
      melody_rnn_generate.get_steps_per_beat(),
      melody_rnn_generate.get_hparams()) as generator:
    melody_rnn_generate.run_with_flags(generator)


if __name__ == '__main__':
  tf.app.run()

