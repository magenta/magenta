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
r"""Create a dataset for training and evaluating the attention RNN model.

Example usage:
  $ bazel build magenta/models/attention_rnn:attention_rnn_create_dataset

  $ ./bazel-bin/magenta/models/attention_rnn/attention_rnn_create_dataset \
    --input=/tmp/note_sequences.tfrecord \
    --output_dir=/tmp/attention_rnn
    --eval_ratio=0.10

See /magenta/models/shared/melody_rnn_create_dataset.py for flag descriptions.
"""

# internal imports
import tensorflow as tf

from magenta.models.attention_rnn import attention_rnn_encoder_decoder
from magenta.models.shared import melody_rnn_create_dataset


def get_pipeline():
  return melody_rnn_create_dataset.get_pipeline(
      attention_rnn_encoder_decoder.MelodyEncoderDecoder())


def main(unused_argv):
  melody_rnn_create_dataset.run_from_flags(get_pipeline())


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
