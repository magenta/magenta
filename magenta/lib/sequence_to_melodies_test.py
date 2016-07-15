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
"""Tests for converting NoteSequence to TensorFlow's SequenceExamples."""

import os
import os.path
import tempfile

# internal imports
import tensorflow as tf

from magenta.lib import melodies_lib
from magenta.lib import sequence_to_melodies

NUM_SPECIAL_EVENTS = melodies_lib.NUM_SPECIAL_EVENTS


class OneHotEncoderDecoder(melodies_lib.MelodyEncoderDecoder):

  def __init__(self):
    super(OneHotEncoderDecoder, self).__init__()
    self._input_size = self.max_note - self.min_note + NUM_SPECIAL_EVENTS
    self._num_classes = self.max_note - self.min_note + NUM_SPECIAL_EVENTS

  @property
  def input_size(self):
    return self._input_size

  @property
  def num_classes(self):
    return self._num_classes

  def melody_to_input(self, melody):
    input_ = [0.0] * self._input_size
    index = (melody.events[-1] + NUM_SPECIAL_EVENTS if melody.events[-1] < 0
             else melody.events[-1] - self.min_note + NUM_SPECIAL_EVENTS)
    input_[index] = 1.0
    return input_

  def melody_to_label(self, melody):
    return (melody.events[-1] + NUM_SPECIAL_EVENTS if melody.events[-1] < 0
            else melody.events[-1] - self.min_note + NUM_SPECIAL_EVENTS)

  def class_index_to_melody_event(self, class_index, melody):
    return (class_index - NUM_SPECIAL_EVENTS if class_index < NUM_SPECIAL_EVENTS
            else class_index + self.min_note - NUM_SPECIAL_EVENTS)


class SequenceToMelodiesTest(tf.test.TestCase):

  def setUp(self):
    self.sequences_file = os.path.join(
        tf.resource_loader.get_data_files_path(),
        '../testdata/notesequences.tfrecord')
    self.tmp_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    self.train_output=os.path.join(self.tmp_dir, 'train_samples.tfrecord')
    self.eval_output=os.path.join(self.tmp_dir, 'eval_samples.tfrecord')
    self.melody_encoder_decoder = OneHotEncoderDecoder()

  def testRunConversion(self):
    sequence_to_melodies.run_conversion(
        melody_encoder_decoder=self.melody_encoder_decoder,
        note_sequences_file=self.sequences_file,
        train_output=self.train_output, eval_output=self.eval_output,
        eval_ratio=0.25)

    self.assertTrue(os.path.isfile(self.train_output))
    reader = tf.python_io.tf_record_iterator(self.train_output)
    num_train_samples = len(list(reader))

    self.assertTrue(os.path.isfile(self.eval_output))
    reader = tf.python_io.tf_record_iterator(self.eval_output)
    num_eval_samples = len(list(reader))

    self.assertTrue(num_train_samples > 0)
    self.assertTrue(num_eval_samples > 0)
    self.assertTrue(num_train_samples > num_eval_samples)

  def testRunConversionNoEval(self):
    sequence_to_melodies.run_conversion(
        melody_encoder_decoder=self.melody_encoder_decoder,
        note_sequences_file=self.sequences_file,
        train_output=self.train_output)

    self.assertTrue(os.path.isfile(self.train_output))
    reader = tf.python_io.tf_record_iterator(self.train_output)
    self.assertEqual(40, len(list(reader)))

    self.assertFalse(os.path.isfile(self.eval_output))


if __name__ == '__main__':
  tf.test.main()
