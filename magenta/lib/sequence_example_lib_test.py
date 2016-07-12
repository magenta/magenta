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
"""Tests for encoders."""

# internal imports
import tensorflow as tf

from magenta.lib import melodies_lib
from magenta.lib import sequence_example_lib

NO_EVENT = melodies_lib.NO_EVENT
NOTE_OFF = melodies_lib.NOTE_OFF


def make_sequence_example(inputs, labels):
  input_features = [
      tf.train.Feature(float_list=tf.train.FloatList(value=input_))
      for input_ in inputs]
  label_features = [
      tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
      for label in labels]
  feature_list = {
      'inputs': tf.train.FeatureList(feature=input_features),
      'labels': tf.train.FeatureList(feature=label_features)
  }
  feature_lists = tf.train.FeatureLists(feature_list=feature_list)
  return tf.train.SequenceExample(feature_lists=feature_lists)


def one_hot(value, length):
  return [1.0 if value == i else 0.0 for i in range(length)]


class CreateDatasetTest(tf.test.TestCase):

  def testBasicOneHotEncoder(self):
    min_note = 48
    max_note = 84
    num_classes = max_note - min_note + 2

    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list(
        [NO_EVENT, 60, 62, 64, NO_EVENT, NOTE_OFF, 65, 67, NOTE_OFF, 69, 71, 72,
         NO_EVENT, NOTE_OFF, 74, 76, 77, 79, NO_EVENT, NOTE_OFF])
    transformed_melody = [NO_EVENT, 12, 14, 16, NO_EVENT, NOTE_OFF, 17, 19,
                          NOTE_OFF, 21, 23, 24, NO_EVENT, NOTE_OFF, 26, 28,
                          29, 31, NO_EVENT, NOTE_OFF]
    expected_inputs = ([one_hot(note + 2, num_classes)
                        for note in transformed_melody] +
                       [one_hot(0, num_classes)] * 12)
    expected_labels = [note + 2 for note in transformed_melody[1:]] + [0] * 13
    expected_sequence_example = make_sequence_example(expected_inputs,
                                                      expected_labels)
    sequence_example = sequence_example_lib.one_hot_encoder(
        melody, min_note, max_note)
    self.assertEqual(expected_sequence_example, sequence_example)

  def testBasicOneHotEncoderTruncateNoteOff(self):
    min_note = 48
    max_note = 84
    num_classes = max_note - min_note + 2

    melody = melodies_lib.MonophonicMelody()
    melody.from_event_list(
        [NO_EVENT, 60, 62, 64, NO_EVENT, NOTE_OFF, 65, 67, NOTE_OFF, 69, 71, 72,
         NO_EVENT, NOTE_OFF, 74, 76, NOTE_OFF])
    transformed_melody = [NO_EVENT, 12, 14, 16, NO_EVENT, NOTE_OFF, 17, 19,
                          NOTE_OFF, 21, 23, 24, NO_EVENT, NOTE_OFF, 26, 28]
    expected_inputs = [one_hot(note + 2, num_classes)
                       for note in transformed_melody]
    expected_labels = ([note + 2 for note in transformed_melody[1:]] +
                       [NOTE_OFF + 2])
    expected_sequence_example = make_sequence_example(expected_inputs,
                                                      expected_labels)
    sequence_example = sequence_example_lib.one_hot_encoder(
        melody, min_note, max_note)
    self.assertEqual(expected_sequence_example, sequence_example)


if __name__ == '__main__':
  tf.test.main()
