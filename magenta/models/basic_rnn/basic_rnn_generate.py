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
r"""Generate melodies from a trained checkpoint of the basic RNN model.

Example usage:
  $ bazel build magenta/models/basic_rnn:basic_rnn_generate

  $ ./bazel-bin/magenta/models/basic_rnn/basic_rnn_generate \
    --run_dir=/tmp/lookback_rnn/logdir/run1 \
    --output_dir=/tmp/lookback_rnn/generated \
    --num_outputs=10 \
    --num_steps=128 \
    --primer_melody="[60]"

See /magenta/models/shared/melody_rnn_generate.py for flag descriptions.
"""

# internal imports
import tensorflow as tf

from magenta.models.basic_rnn import basic_rnn_generator
from magenta.models.shared import melody_rnn_generate


def main(unused_argv):
  melody_rnn_generate.setup_logs()

  with basic_rnn_generator.create_generator(
      melody_rnn_generate.get_checkpoint(),
      melody_rnn_generate.get_bundle(),
      melody_rnn_generate.get_steps_per_quarter(),
      melody_rnn_generate.get_hparams()) as generator:
    if melody_rnn_generate.should_save_generator_bundle():
      tf.logging.info('Saving generator bundle to %s' % (
          melody_rnn_generate.get_bundle_file()))
      generator.create_bundle_file(melody_rnn_generate.get_bundle_file())
    else:
      melody_rnn_generate.run_with_flags(generator)


def console_entry_point():
  tf.app.run(main)

if __name__ == '__main__':
  console_entry_point()
