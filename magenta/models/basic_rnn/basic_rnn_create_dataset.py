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
r"""Create a dataset for training and evaluating the basic RNN model.

Example usage:
  $ bazel build magenta/models/basic_rnn:basic_rnn_create_dataset

  $ ./bazel-bin/magenta/models/basic_rnn/basic_rnn_create_dataset \
    --input=/tmp/note_sequences.tfrecord \
    --train_output=/tmp/basic_rnn/training_melodies.tfrecord \
    --eval_output=/tmp/basic_rnn/eval_melodies.tfrecord \
    --eval_ratio=0.10

See /magenta/models/shared/melody_rnn_create_dataset.py for flag descriptions.
"""


# internal imports
import tensorflow as tf

from magenta.lib import melodies_lib
from magenta.models.shared import melody_rnn_create_dataset
from magenta.models.basic_rnn import basic_rnn_encoder_decoder
from magenta.pipelines import pipeline


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', None,
                           'TFRecord to read NoteSequence protos from.')
tf.app.flags.DEFINE_string('output_dir', None,
                           'Directory to write training and eval TFRecord '
                           'files. The TFRecord files are populated with '
                           'SequenceExample protos.')
tf.app.flags.DEFINE_float('eval_ratio', 0.0,
                          'Fraction of input to set asside for eval set. '
                          'Partition is randomly selected.')


def main(unused_argv):
  melody_encoder_decoder = basic_rnn_encoder_decoder.MelodyEncoderDecoder()
  pipeline_instance = melody_rnn_create_dataset.BasicRNNPipeline(
      melody_encoder_decoder, FLAGS.eval_ratio)
  pipeline.run_pipeline_serial(
      pipeline_instance,
      pipeline.tf_record_iterator(FLAGS.input, pipeline_instance.input_type),
      FLAGS.output_dir)


if __name__ == '__main__':
  tf.app.run()
