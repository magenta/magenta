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
"""Create basic RNN dataset from NoteSequence protos.

This script will extract melodies from NoteSequence protos and save them to
TensorFlow's SequenceExample protos for input to the basic RNN model.
"""

import logging
import sys
import numpy as np
import random

# internal imports
import tensorflow as tf

from magenta.lib import melodies_lib
from magenta.pipelines import pipeline
from magenta.pipelines import modules
from magenta.protobuf import music_pb2


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


class OneHotEncoder(modules.Module):
  input_type = melodies_lib.Melody
  output_type = tf.train.SequenceExample

  def __init__(self, min_note=48, max_note=84, transpose_to_key=0):
    super(OneHotEncoder, self).__init__()
    self.min_note = min_note
    self.max_note = max_note
    self.transpose_to_key = transpose_to_key

  def transform(self, melody):
    """Converts a melody into a list of input features and a list of labels.

    This encoder converts each melody note to a one-hot vector (a list of floats
    that are all 0.0 and 1.0 at the index equal to the encoded value). The one-hot
    length is `max_note` - `min_note` + 2. NO_EVENT gets 0th position. NOTE_OFF
    gets 1st position. Pitches get pitch + 2.

    Two tensors are created: model inputs and model labels. Inputs are one-hot
    vectors. Each label is the note index (an int) of the one_hot that comes next.
    The vector of labels is equal to the vector of inputs (without one-hot
    encoding) shifted left by 1 and padded with a NO_EVENT or NOTE_OFF.

    The intput and label sequence lengths are padded with NO_EVENT to a multiple
    of 4 * `steps_per_beat` to make them end at the end of a bar. Final bars with
    only a single event that is a NOTE_OFF are truncated rather than padded.

    Args:
      melody: A Melody object to encode.
      steps_per_beat: Number of subdivisions of each beat. 4/4 time is assumed, so
          steps per bar is 4 * `steps_per_beat`.
      min_note: Minimum pitch (inclusive) that the output notes will take on.
      max_note: Maximum pitch (exclusive) that the output notes will take on.
      transpose_to_key: The melody is transposed to be in this key. 0 = C Major.

    Returns:
      sequence_example: A SequenceExample proto containing inputs and labels sequences.
      reconstruction_data: Information needed to decode encoder output.
    """

    transpose_amount = melody.squash(self.min_note, self.max_note, self.transpose_to_key)
    note_range = self.max_note - self.min_note
    one_hot_length = note_range + melodies_lib.NUM_SPECIAL_EVENTS
    note_indices = [
        note + melodies_lib.NUM_SPECIAL_EVENTS if note < 0
        else note - self.min_note + melodies_lib.NUM_SPECIAL_EVENTS
        for note in melody]
    inputs = np.zeros((len(note_indices), one_hot_length), dtype=float)
    inputs[np.arange(len(note_indices)), note_indices] = 1.0
    labels = (note_indices[1:] +
              [melodies_lib.NO_EVENT + melodies_lib.NUM_SPECIAL_EVENTS])

    # Pad to the end of the measure.
    if len(inputs) % melody.steps_per_bar == 1:
      # Last event is always note off. If that is the only event in the last bar,
      # remove it.
      inputs = inputs[:-1]
      labels = labels[:-1]
    elif len(inputs) % melody.steps_per_bar > 1:
      # Pad out to end of the bar.
      pad_len = melody.steps_per_bar - len(inputs) % melody.steps_per_bar
      padding = np.zeros((pad_len, one_hot_length), dtype=float)
      padding[:, 0] = 1.0
      inputs = np.concatenate((inputs, padding), axis=0)
      labels += [0] * pad_len

    input_features = [tf.train.Feature(float_list=tf.train.FloatList(value=input_))
                      for input_ in inputs]
    label_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                      for label in labels]
    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features),
        'labels': tf.train.FeatureList(feature=label_features)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return [tf.train.SequenceExample(feature_lists=feature_lists)]


def random_partition(input_list, partition_ratio):
  partitions = [], []
  for item in input_list:
    partition_index = int(random.random() < partition_ratio)
    partitions[partition_index].append(item)
  return partitions  # old, new


class BasicRNNPipeline(pipeline.Pipeline):
  input_type = music_pb2.NoteSequence
  output_type = tf.train.SequenceExample

  def __init__(self, eval_ratio):
    super(BasicRNNPipeline, self).__init__()
    self.output_names = ['basic_rnn_train', 'basic_rnn_eval']
    self.eval_ratio = eval_ratio
    self.quantizer = modules.Quantizer(steps_per_beat=4)
    self.melody_extractor = modules.MelodyExtractor(min_bars=7, min_unique_pitches=5, gap_bars=1.0)
    self.one_hot_encoder = OneHotEncoder()

  def transform(self, input):
    outputs = self.quantizer.transform(input)
    outputs = [output for input in outputs for output in self.melody_extractor.transform(input)]
    outputs = [output for input in outputs for output in self.one_hot_encoder.transform(input)]
    train_set, eval_set = random_partition(outputs, self.eval_ratio)
    return {self.output_names[0]: train_set, self.output_names[1]: eval_set}

  def get_stats(self):
    return {}

  def get_output_names(self):
    return self.output_names


def main(unused_argv):
  root = logging.getLogger()
  root.setLevel(logging.INFO)
  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.INFO)
  root.addHandler(ch)

  pipeline_instance = BasicRNNPipeline(FLAGS.eval_ratio)
  pipeline.run_pipeline_serial(pipeline_instance, pipeline.TFRecordIterator(FLAGS.input, pipeline_instance.input_type), FLAGS.output_dir)


if __name__ == '__main__':
  tf.app.run()
