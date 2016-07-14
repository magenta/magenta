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
    --train_output=/tmp/attention_rnn/training_melodies.tfrecord \
    --eval_output=/tmp/attention_rnn/eval_melodies.tfrecord \
    --eval_ratio=0.10

See /magenta/models/shared/melody_rnn_create_dataset.py for flag descriptions.
"""

# internal imports
import attention_rnn_encoder_decoder
import tensorflow as tf

from magenta.models.shared import melody_rnn_create_dataset


def main(unused_argv):
  melody_encoder_decoder = attention_rnn_encoder_decoder.MelodyEncoderDecoder()
  melody_rnn_create_dataset.run(melody_encoder_decoder)


if __name__ == '__main__':
  tf.app.run()
