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
"""Create a dataset of SequenceExamples from NoteSequence protos.

This script will extract melodies from NoteSequence protos and save them to
TensorFlow's SequenceExample protos for input to the melody RNN models.
"""

import logging
import os
import sys

# internal imports
import tensorflow as tf

from magenta.lib import sequence_to_melodies


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', None,
                           'TFRecord to read NoteSequence protos from.')
tf.app.flags.DEFINE_string('train_output', None,
                           'TFRecord to write SequenceExample protos to. '
                           'Contains training set.')
tf.app.flags.DEFINE_string('eval_output', None,
                           'TFRecord to write SequenceExample protos to. '
                           'Contains eval set. No eval set is produced if '
                           'this flag is not set.')
tf.app.flags.DEFINE_float('eval_ratio', 0.0,
                          'Fraction of input to set aside for eval set. '
                          'Partition is randomly selected.')


def run(melody_encoder_decoder):
  """Creates training and eval data with the given MelodyEncoderDecoder.

  Args:
    melody_encoder_decoder: A melodies_lib.MelodyEncoderDecoder.
  """
  root = logging.getLogger()
  root.setLevel(logging.INFO)
  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.INFO)
  root.addHandler(ch)

  train_output_dir = os.path.dirname(FLAGS.train_output)
  if not os.path.exists(train_output_dir):
    os.makedirs(train_output_dir)

  eval_output_dir = os.path.dirname(FLAGS.eval_output)
  if FLAGS.eval_output and not os.path.exists(eval_output_dir):
    os.makedirs(eval_output_dir)

  sequence_to_melodies.run_conversion(melody_encoder_decoder,
                                      FLAGS.input,
                                      FLAGS.train_output,
                                      FLAGS.eval_output,
                                      FLAGS.eval_ratio)
