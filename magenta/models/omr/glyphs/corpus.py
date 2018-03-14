"""Labeled glyph corpus.

Reads Examples holding image patches and glyph records from TFRecords.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np
import tensorflow as tf


def get_patch_shape(corpus_file):
  """Gets the patch shape (height, width) from the corpus file.

  Args:
    corpus_file: Path to a TFRecords file.

  Returns:
    A tuple (height, width), extracted from the first record.

  Raises:
    ValueError: if the corpus_file is empty.
  """
  example = tf.train.Example()
  try:
    example.ParseFromString(next(tf.python_io.tf_record_iterator(corpus_file)))
  except StopIteration as e:
    raise ValueError('corpus_file cannot be empty: %s' % e)
  return (example.features.feature['height'].int64_list.value[0],
          example.features.feature['width'].int64_list.value[0])


def parse_corpus(corpus_file, height, width):
  """Returns tensors holding the parsed result of the corpus file.

  Uses the default TensorFlow session to read examples.

  Args:
    corpus_file: Path to a TFRecords file.
    height: Patch height, as returned from `get_patch_shape`.
    width: Patch width, as returned from `get_patch_width`.

  Returns:
    patches: float32 tensor with shape (num_patches, height, width).
    labels: int64 tensor with shape (num_patches,).
  """
  sess = tf.get_default_session()
  producer = tf.train.string_input_producer([corpus_file], num_epochs=1)
  unused_keys, examples = tf.TFRecordReader().read_up_to(producer, 10000)
  parsed_examples = tf.parse_example(examples, {
      'patch': tf.FixedLenFeature((height, width), tf.float32),
      'label': tf.FixedLenFeature((), tf.int64)
  })
  sess.run(tf.local_variables_initializer())  # initialize num_epochs
  coord = tf.train.Coordinator()
  queue_runners = tf.train.start_queue_runners(start=True, coord=coord)
  assert queue_runners, 'started queue runners'
  all_patches = []
  all_labels = []
  while True:
    try:
      patch, label = sess.run(
          [parsed_examples['patch'], parsed_examples['label']])
    except tf.errors.OutOfRangeError:
      break  # done
    all_patches.append(patch)
    all_labels.append(label)
  coord.request_stop()
  coord.join()
  return np.concatenate(all_patches), np.concatenate(all_labels)
