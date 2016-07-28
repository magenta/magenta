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

import abc
import inspect
import os.path

# internal imports
import tensorflow as tf

from magenta.pipelines import statistics


class BadTypeSignatureException(Exception):
  pass


class Key(object):
  """Represents a get operation on a Pipeline type signature.

  If a pipeline instance `my_pipeline` has `output_type`
  {'key_1': Type1, 'key_2': Type2}, then Key(my_pipeline, 'key_1'),
  represents the output type Type1. And likewise Key(my_pipeline, 'key_2')
  represents Type2.

  Calling __getitem__ on a pipeline will return a Key instance.
  So my_pipeline['key_1'] returns Key(my_pipeline, 'key_1'), and so on.

  Key objects are used for assembling a directed asyclic graph of Pipeline
  instances. See dag_pipeline.py.
  """

  def __init__(self, unit, key, type_):
    if not isinstance(unit.output_type, dict):
      raise KeyError(
          'Cannot take key %s of %s because output type %s is not a dictionary'
          % (key, unit, unit.output_type))
    if key not in unit.output_type:
      raise KeyError('Key %s is not valid for %s with output type %s'
                     % (key, unit, unit.output_type))
    self.key = key
    self.unit = unit
    self.output_type = type_

  def __repr__(self):
    return 'Key(%s, %s)' % (self.unit, self.key)


def _guarantee_dict(given, default_name):
  if not isinstance(given, dict):
    return {default_name: dict}
  return given


def _assert_valid_type_signature(type_sig, type_sig_name):
  if isinstance(type_sig, dict):
    for k, val in type_sig.items():
      if not isinstance(k, basestring):
        raise BadTypeSignatureException(
            '%s key %s must be a string.' % (type_sig_name, k))
      if not inspect.isclass(val):
        raise BadTypeSignatureException(
            '%s %s at key %s must be a Python class.' % (type_sig_name, val, k))
  else:
    if not inspect.isclass(type_sig):
      raise BadTypeSignatureException(
          '%s %s must be a Python class.' % (type_sig_name, type_sig))


class Pipeline(object):
  """An abstract class for data processing pipelines that transform datasets.

  A Pipeline can transform one or many inputs to one or many outputs. When there
  are many inputs or outputs, each input/output is assigned a string name.

  The `transform` method converts a given input or dictionary of inputs to
  a list of transformed outputs, or a dictionary mapping names to lists of
  transformed outputs for each name.

  The `get_stats` method returns any statistics that were collected during the
  last call to `transform`. These statistics can give feedback about why any
  data was discarded and what the input data is like.
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, input_type, output_type):
    # Make sure `input_type` and `output_type` are valid.
    _assert_valid_type_signature(input_type, 'input_type')
    _assert_valid_type_signature(output_type, 'output_type')
    self._input_type = input_type
    self._output_type = output_type

  def __getitem__(self, key):
    return Key(self, key, self.output_type_as_dict[key])

  @property
  def input_type(self):
    """What type or types does this pipeline take as input.

    Returns:
      A class, or a dictionary mapping names to classes.
    """
    return self._input_type

  @property
  def output_type(self):
    """What type or types does this pipeline output.

    Returns:
      A class, or a dictionary mapping names to classes.
    """
    return self._output_type

  @property
  def output_type_as_dict(self):
    """Returns a dictionary mapping names to classes.

    If `output_type` is a single class, then a default name will be created
    for the output and a dictionary containing `output_type` will be returned.

    Returns:
      Dictionary mapping names to output types.
    """
    return _guarantee_dict(self._output_type, 'dataset')

  @abc.abstractmethod
  def transform(self, input_object):
    """Runs the pipeline on the given input.

    Args:
      input_object: An object or dictionary mapping names to objects.
          The object types must match `input_type`.

    Returns:
      If `output_type` is a class, `transform` returns a list of objects
      which are all that type. If `output_type` is a dictionary mapping
      names to classes, `transform` returns a dictionary mapping those
      same names to lists of objects that are the type mapped to each name.
    """
    pass

  def get_stats(self):
    """Returns statistics about pipeline runs.

    Call `get_stats` after each call to `transform`.
    `transform` computes statistics which will be returned here.

    Returns:
      Dictionary mapping statistic name to statistic value.
    """
    return {}


def file_iterator(root_dir, extension=None, recurse=True):
  """Generator that iterates over all files in the given directory.

  Will recurse into sub-directories if `recurse` is True.

  Args:
    root_dir: Path to root directory to search for files in.
    extension: If given, only files with the given extension are opened.
    recurse: If True, subdirectories will be traversed. Otherwise, only files
        in `root_dir` are opened.

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
  dirs = [os.path.join(root_dir, child)
          for child in tf.gfile.ListDirectory(root_dir)]
  while dirs:
    sub = dirs.pop()
    if tf.gfile.IsDirectory(sub):
      if recurse:
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

  Run the the pipeline on each input from the iterator one at a time.
  A file will be written to `output_dir` for each dataset name specified
  by the pipeline. pipeline.transform is called on each input and the
  results are aggregated into their correct datasets.

  The output type or types given by `pipeline.output_type` must be protocol
  buffers or objects that have a SerializeToString method.

  Args:
    pipeline: A Pipeline instance. `pipeline.output_type` must be a protocol
        buffer or a dictionary mapping names to protocol buffers.
    input_iterator: Iterates over the input data. Items returned by it are fed
        directly into the pipeline's `transform` method.
    output_dir: Path to directory where datasets will be written. Each dataset
        is a file whose name contains the pipeline's dataset name. If the
        directory does not exist, it will be created.


  Raises:
    ValueError: If any of `pipeline`'s output types do not have a
        SerializeToString method.
  """
  if isinstance(pipeline.output_type, dict):
    for name, type_ in pipeline.output_type.items():
      if not hasattr(type_, 'SerializeToString'):
        raise ValueError(
            'Pipeline output "%s" does not have method SerializeToString. '
            'Output type = %s' % (name, pipeline.output_type))
  else:
    if not hasattr(pipeline.output_type, 'SerializeToString'):
      raise ValueError(
          'Pipeline output type %s does not have method SerializeToString.'
          % pipeline.output_type)

  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  output_names = pipeline.output_type_as_dict.keys()

  output_paths = [os.path.join(output_dir, name + '.tfrecord')
                  for name in output_names]
  writers = dict([(name, tf.python_io.TFRecordWriter(path))
                  for name, path in zip(output_names, output_paths)])

  total_inputs = 0
  total_outputs = 0
  stats = {}
  for input_ in input_iterator:
    total_inputs += 1
    for name, outputs in _guarantee_dict(pipeline.transform(input_),
                                         output_names[0]).items():
      for output in outputs:
        writers[name].write(output.SerializeToString())
        total_outputs += 1
    statistics.merge_statistics_dicts(stats, pipeline.get_stats())
    if total_inputs % 500 == 0:
      tf.logging.info('Processed %d inputs so far. Produced %d outputs. '
                      'Statistics:', total_inputs, total_outputs)
      statistics.log_statistics_dict(stats, tf.logging.info)
  tf.logging.info('Completed.')
  tf.logging.info('Processed %d inputs total. Produced %d outputs. '
                  'Statistics:', total_inputs, total_outputs)
  statistics.log_statistics_dict(stats, tf.logging.info)


def load_pipeline(pipeline, input_iterator):
  """Runs a pipeline saving the output into memory.

  Use this instead of `run_pipeline_serial` to build a dataset on the fly
  without saving it to disk.

  Args:
    pipeline: A Pipeline instance.
    input_iterator: Iterates over the input data. Items returned by it are fed
        directly into the pipeline's `transform` method.

  Returns:
    The aggregated return values of pipeline.transform. Specifically a
    dictionary mapping dataset names to lists of objects. Each name acts
    as a bucket where outputs are aggregated.
  """
  aggregated_outputs = dict(
      [(name, []) for name in pipeline.output_type_as_dict])
  for input_object in input_iterator:
    outputs = _guarantee_dict(pipeline.transform(input_object),
                              aggregated_outputs.keys()[0])
    for name, output_list in outputs.items():
      aggregated_outputs[name].extend(output_list)
  return aggregated_outputs
