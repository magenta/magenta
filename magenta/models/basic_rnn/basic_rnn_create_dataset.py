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
import random
import sys

# internal imports
import tensorflow as tf

from magenta.lib import melodies_lib
from magenta.models.basic_rnn import basic_rnn_ops
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
  input_type = melodies_lib.MonophonicMelody
  output_type = tf.train.SequenceExample

  def __init__(self, min_note=48, max_note=84, transpose_to_key=0):
    super(OneHotEncoder, self).__init__()
    self.min_note = min_note
    self.max_note = max_note
    self.transpose_to_key = transpose_to_key

  def transform(self, melody):
    _ = melody.squash(self.min_note, self.max_note, self.transpose_to_key)
    encoded = basic_rnn_ops.one_hot_encoder(melody)
    return [encoded]


def random_partition(input_list, partition_ratio):
  partitions = [], []
  for item in input_list:
    partition_index = int(random.random() < partition_ratio)
    partitions[partition_index].append(item)
  return partitions  # old, new


def update_dict_with_prefix(update_dict, merge_dict, prefix):
  prefix_dict = dict([(prefix + k, v) for k, v in merge_dict.items()])
  update_dict.update(prefix_dict)

class BasicRNNPipeline(pipeline.Pipeline):
  input_type = music_pb2.NoteSequence
  output_type = tf.train.SequenceExample

  def __init__(self, eval_ratio):
    super(BasicRNNPipeline, self).__init__()
    self.output_names = ['basic_rnn_train', 'basic_rnn_eval']
    self.eval_ratio = eval_ratio
    self.quantizer = modules.Quantizer(steps_per_beat=4)
    self.melody_extractor = modules.MonophonicMelodyExtractor(min_bars=7, min_unique_pitches=5, gap_bars=1.0)
    self.one_hot_encoder = OneHotEncoder()
    self.stats_dict = {}

  def transform(self, input):
    outputs = self.quantizer.transform(input)
    outputs = [output for input in outputs for output in self.melody_extractor.transform(input)]
    outputs = [output for input in outputs for output in self.one_hot_encoder.transform(input)]
    train_set, eval_set = random_partition(outputs, self.eval_ratio)

    self.stats_dict = {}
    update_dict_with_prefix(self.stats_dict, self.quantizer.get_stats(), type(self.quantizer).__name__ + "_")
    update_dict_with_prefix(self.stats_dict, self.melody_extractor.get_stats(), type(self.melody_extractor).__name__ + "_")
    update_dict_with_prefix(self.stats_dict, self.one_hot_encoder.get_stats(), type(self.one_hot_encoder).__name__ + "_")

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
  pipeline.run_pipeline_serial(
      pipeline_instance,
      pipeline.tf_record_iterator(FLAGS.input, pipeline_instance.input_type),
      FLAGS.output_dir)


if __name__ == '__main__':
  tf.app.run()
