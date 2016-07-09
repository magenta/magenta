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
"""For running data processing pipelines."""

import logging
import os.path

# internal imports
import tensorflow as tf


class Pipeline(object):

  def transform(self, input):
    # return dict mapping output descriptor to list of outputs
    pass

  def get_stats(self):
    return {}

  def get_output_names(self):
    return []


def RawFileIterator(midi_directory):
  # TODO: generator over files in directory
  # yield raw bytes for each file
  pass


# TODO: are there sharded TFRecords?
def TFRecordIterator(tfrecord_file, proto):
  for raw_bytes in tf.python_io.tf_record_iterator(tfrecord_file):
    yield proto.FromString(raw_bytes)


def PipelineIterator(pipeline, input_iterator):
  for input in input_iterator:
    outputs = pipeline.transform(input)
    for output in outputs:
      yield output


# TODO: are there sharded TFRecords?
def run_pipeline_serial(pipeline, input_iterator, output_dir):
  output_names = pipeline.get_output_names()

  output_paths = [os.path.join(output_dir, name + '.tfrecord') for name in output_names]
  writers = dict([(name, tf.python_io.TFRecordWriter(path)) for name, path in zip(output_names, output_paths)])

  total_inputs = 0
  total_outputs = 0
  for input in input_iterator:
    total_inputs += 1
    for name, outputs in pipeline.transform(input).items():
      for output in outputs:
        writers[name].write(output.SerializeToString())
        total_outputs += 1
    if total_inputs % 10 == 0:
      logging.info('%d inputs. %d outputs. stats = %s' % (total_inputs, total_outputs, pipeline.get_stats()))