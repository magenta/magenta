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
  """Base class for data processing pipelines that transform datasets."""

  def transform(self, input_object):
    """Runs the pipeline on the given input.

    Subclasses must implement this method.

    Args:
      input_object: Any object. The required type depends on implementation.

    Returns:
      A dictionary mapping output names to lists of objects. The object types
      depend on implementation. Each output name corresponds to an output
      collection. See get_output_names method.
    """
    pass

  def get_stats(self):
    """Returns statistics about pipeline runs.

    Call after running transform to get statistics about it.

    Returns:
      Dictionary mapping statistic name to statistic value.
    """
    return {}

  def get_output_names(self):
    """Return output names.

    An output name is the name of a dataset. The user of this Pipeline
    instance decides where to save that dataset. Pipeline.transform
    outputs lists for any of the output names returned here. Its up to the
    user to aggregate everything into the correct datasets.

    Returns:
      List of strings, where each string is a dataset name.
    """
    return []


def recursive_file_iterator(root_dir, extension=None):
  """Generator that iterates over all files in the given directory.

  Will recurse into sub-directories.

  Args:
    root_dir: Path to root directory to search for files in.
    extension: If given, only files with the given extension are opened.

  Yields:
    Raw bytes (as a string) of each file opened.

  Raises:
    ValueError: When extension is an empty string. Leave as None to omit.
  """
  if extension is not None:
    if not extension:
      raise ValueError('File extension cannot be an empty string.')
    extension = extension.lower()
    if extension[0] != '.':
      extension = '.' + extension
  dirs = [root_dir]
  while dirs:
    sub = dirs.pop()
    if tf.gfile.IsDirectory(sub):
      dirs.extend(
          [os.path.join(sub, child) for child in tf.gfile.ListDirectory(sub)])
    else:
      if extension is None or sub.lower().endswith(extension):
        with open(sub, 'rb') as f:
          yield f.read()


def tf_record_iterator(tfrecord_file, proto):
  """Generator that iterates over protocol buffers in a TFRecord file.

  Args:
    tfrecord_file: Path to a TFRecord file containing protocol buffers.
    proto: A protocol buffer class. This type will be used to deserialize the
        protos from the TFRecord file. This will be the output type.

  Yields:
    Instances of the given `proto` class from the TFRecord file.
  """
  for raw_bytes in tf.python_io.tf_record_iterator(tfrecord_file):
    yield proto.FromString(raw_bytes)


def run_pipeline_serial(pipeline, input_iterator, output_dir):
  """Runs the a pipeline on a data source and writes to a directory.

  Will the the pipeline on each input from the iterator one at a time.
  A file will be written to `output_dir` for each dataset name specified
  by the pipeline. pipeline.transform is called on each input and the
  results are aggregated into their correct datasets.

  Args:
    pipeline: A Pipeline instance.
    input_iterator: Iterates over the input data. Items returned by it are fed
        directly into the pipeline's `transform` method.
    output_dir: Path to directory where datasets will be written. Each dataset
        is a file whose name contains the pipeline's dataset name.
  """
  output_names = pipeline.get_output_names()

  output_paths = [os.path.join(output_dir, name + '.tfrecord')
                  for name in output_names]
  writers = dict([(name, tf.python_io.TFRecordWriter(path))
                  for name, path in zip(output_names, output_paths)])

  total_inputs = 0
  total_outputs = 0
  for input_ in input_iterator:
    total_inputs += 1
    for name, outputs in pipeline.transform(input_).items():
      for output in outputs:
        writers[name].write(output.SerializeToString())
        total_outputs += 1
    if total_inputs % 10 == 0:
      logging.info('%d inputs. %d outputs. stats = %s',
                   total_inputs, total_outputs, pipeline.get_stats())


def pipeline_iterator(pipeline, input_iterator):
  """Generator that runs a pipeline.

  Use this instead of `run_pipeline_serial` to build a dataset on the fly
  without saving it to disk.

  Args:
    pipeline: A Pipeline instance.
    input_iterator: Iterates over the input data. Items returned by it are fed
        directly into the pipeline's `transform` method.

  Yields:
    The return values of pipeline.transform. Specifically a dictionary
    mapping dataset names to lists of objects.
  """
  for input_ in input_iterator:
    outputs = pipeline.transform(input_)
    yield outputs
