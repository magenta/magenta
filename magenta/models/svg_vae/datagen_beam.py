# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Beam pipelines to generate examples for the GlyphAzzn dataset."""
from absl import app
from absl import flags

import apache_beam as beam
from magenta.models.svg_vae import svg_utils
import numpy as np
from tensor2tensor.data_generators import generator_utils
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'pipeline_options', '',
    'Command line flags to use in constructing the Beam pipeline options.')
flags.DEFINE_string(
    'raw_data_file', '/path/to/parquetio-file',
    'File where the raw data is (in parquetio format).')
flags.DEFINE_string(
    'final_data_file', '/path/to/final-dataset-train',
    'File where the final data will be saved (in tfrecord format).')
flags.DEFINE_string(
    'final_stats_file', '/path/to/final-dataset-stats',
    'File where the final data stats will be saved (in tfrecord format).')

# pylint: disable=expression-not-assigned
# pylint: disable=abstract-method
# pylint: disable=arguments-differ


################## HELPERS FOR DATASET PROCESSING ###################
def _is_valid_glyph(g):
  is_09 = 48 <= g['uni'] <= 57
  is_capital_az = 65 <= g['uni'] <= 90
  is_az = 97 <= g['uni'] <= 122
  is_valid_dims = g['width'] != 0 and g['vwidth'] != 0
  return (is_09 or is_capital_az or is_az) and is_valid_dims


def _is_valid_path(pathunibfp):
  return pathunibfp[0] and len(pathunibfp[0]) <= 50


def _convert_to_path(g):
  """Converts SplineSet in SFD font to str path."""
  path = svg_utils.sfd_to_path_list(g)
  path = svg_utils.add_missing_cmds(path, remove_zs=False)
  path = svg_utils.normalize_based_on_viewbox(
      path, '0 0 {} {}'.format(g['width'], g['vwidth']))
  return path, g['uni'], g['binary_fp']


def _create_example(pathuni):
  """Bulk of dataset processing. Converts str path to serialized tf.Example."""
  path, uni, binary_fp = pathuni
  final = {}

  # zoom out
  path = svg_utils.zoom_out(path)
  # make clockwise
  path = svg_utils.canonicalize(path)

  # render path for training
  final['rendered'] = svg_utils.per_step_render(path, absolute=True)

  # make path relative
  path = svg_utils.make_relative(path)
  # convert to vector
  vector = svg_utils.path_to_vector(path, categorical=True)
  # make simple vector
  vector = np.array(vector)
  vector = np.concatenate(
      [np.take(vector, [0, 4, 5, 9], axis=-1), vector[..., -6:]], axis=-1)

  # count some stats
  final['seq_len'] = np.shape(vector)[0]
  final['class'] = int(svg_utils.map_uni_to_alphanum(uni))
  final['binary_fp'] = str(binary_fp)

  # append eos
  vector = svg_utils.append_eos(vector.tolist(), True, 10)

  # pad path to 51 (with eos)
  final['sequence'] = np.concatenate(
      (vector, np.zeros(((50 - final['seq_len']), 10))), 0)

  # make pure list:
  final['rendered'] = np.reshape(final['rendered'][..., 0],
                                 [64*64]).astype(np.float32).tolist()
  final['sequence'] = np.reshape(final['sequence'],
                                 [51*10]).astype(np.float32).tolist()
  final['class'] = np.reshape(final['class'],
                              [1]).astype(np.int64).tolist()
  final['seq_len'] = np.reshape(final['seq_len'],
                                [1]).astype(np.int64).tolist()

  return generator_utils.to_example(final).SerializeToString()


def _decode_tfexample(serialized_example):
  """Decodes saved, serialized tfrecord example."""
  eg = tf.train.Example.FromString(serialized_example)
  return {
      # add [0] after "value" if you want to just get value "1" instead of ["1"]
      'class': np.reshape(eg.features.feature['class'].int64_list.value,
                          [1]).astype(np.int64).tolist(),
      'seq_len': eg.features.feature['seq_len'].int64_list.value[0],
      'sequence': np.reshape(eg.features.feature['sequence'].float_list.value,
                             [51*10]).astype(np.float32).tolist(),
      'rendered': np.reshape(eg.features.feature['rendered'].float_list.value,
                             [64*64]).astype(np.float32).tolist(),
  }


def _mean_to_example(mean_stdev):
  """Converts the found mean and stdev to tfrecords example."""
  # mean_stdev is a dict
  mean_stdev['mean'] = np.reshape(mean_stdev['mean'],
                                  [10]).astype(np.float32).tolist()
  mean_stdev['variance'] = np.reshape(mean_stdev['variance'],
                                      [10]).astype(np.float32).tolist()
  mean_stdev['stddev'] = np.reshape(mean_stdev['stddev'],
                                    [10]).astype(np.float32).tolist()
  mean_stdev['count'] = np.reshape(mean_stdev['count'],
                                   [1]).astype(np.int64).tolist()
  return generator_utils.to_example(mean_stdev)


class MeanStddev(beam.CombineFn):
  """Apache Beam accumulator to compute the mean/stdev of svg commands."""

  def create_accumulator(self):
    curr_sum = np.zeros([10])
    sum_sq = np.zeros([10])
    return (curr_sum, sum_sq, 0)  # x, x^2, count

  def add_input(self, sum_count, new_input):
    (curr_sum, sum_sq, count) = sum_count
    # new_input is a dict with keys = ['seq_len', 'sequence']
    new_seq_len = new_input['seq_len']

    # remove padding and eos from sequence
    new_input = np.reshape(np.array(new_input['sequence']), [-1, 10])
    new_input = new_input[:new_seq_len, :]

    # accumulate new_sum and new_sum_sq
    new_sum = np.sum([curr_sum, np.sum(new_input, axis=0)], axis=0)
    new_sum_sq = np.sum([sum_sq, np.sum(np.power(new_input, 2), axis=0)],
                        axis=0)
    return new_sum, new_sum_sq, count + new_seq_len

  def merge_accumulators(self, accumulators):
    curr_sums, sum_sqs, counts = list(zip(*accumulators))
    return np.sum(curr_sums, axis=0), np.sum(sum_sqs, axis=0), np.sum(counts)

  def extract_output(self, sum_count):
    (curr_sum, curr_sum_sq, count) = sum_count
    if count:
      mean = np.divide(curr_sum, count)
      variance = np.divide(curr_sum_sq, count) - np.power(mean, 2)
      # -ve value could happen due to rounding
      variance = np.max([variance, np.zeros(np.shape(variance))], axis=0)
      stddev = np.sqrt(variance)
      return {
          'mean': mean,
          'variance': variance,
          'stddev': stddev,
          'count': count
      }
    else:
      return {
          'mean': float('NaN'),
          'variance': float('NaN'),
          'stddev': float('NaN'),
          'count': 0
      }


########################## PIPELINE GENERATORS ##########################
def create_glyphazzn_dataset(filepattern, output_path):
  """Creates a glyphazzn dataset, from raw Parquetio to TFRecords."""
  def pipeline(root):
    """Pipeline for creating glyphazzn dataset."""
    attrs = ['uni', 'width', 'vwidth', 'sfd', 'id', 'binary_fp']

    examples = root | 'Read' >> beam.io.parquetio.ReadFromParquet(
        file_pattern=filepattern, columns=attrs)

    examples = examples | 'FilterBadIcons' >> beam.Filter(_is_valid_glyph)
    examples = examples | 'ConvertToPath' >> beam.Map(_convert_to_path)
    examples = examples | 'FilterBadPathLenghts' >> beam.Filter(_is_valid_path)
    examples = examples | 'ProcessAndConvert' >> beam.Map(_create_example)
    (examples | 'WriteToTFRecord' >> beam.io.tfrecordio.WriteToTFRecord(
        output_path, num_shards=90))
  return pipeline


def get_stats_of_glyphazzn(filepattern, output_path):
  """Computes the Mean and Std across examples in glyphazzn dataset."""
  def pipeline(root):
    """Pipeline for computing means/std from dataset."""
    examples = root | 'Read' >> beam.io.tfrecordio.ReadFromTFRecord(filepattern)
    examples = examples | 'Deserialize' >> beam.Map(_decode_tfexample)
    examples = examples | 'GetMeanStdev' >> beam.CombineGlobally(MeanStddev())
    examples = examples | 'MeanStdevToSerializedTFRecord' >> beam.Map(
        _mean_to_example)
    (examples | 'WriteToTFRecord' >> beam.io.tfrecordio.WriteToTFRecord(
        output_path, coder=beam.coders.ProtoCode(tf.train.Example)))
  return pipeline


def main(_):
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      FLAGS.pipeline_options.split(','))

  pipeline = create_glyphazzn_dataset(
      FLAGS.raw_data_file + '*',
      FLAGS.final_data_file)
  with beam.Pipeline(options=pipeline_options) as root:
    pipeline(root)

  pipeline = get_stats_of_glyphazzn(
      FLAGS.final_data_file + '*',
      FLAGS.final_stats_file)
  with beam.Pipeline(options=pipeline_options) as root:
    pipeline(root)


if __name__ == '__main__':
  app.run(main)
