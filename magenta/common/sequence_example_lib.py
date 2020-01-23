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

"""Utility functions for working with tf.train.SequenceExamples."""

import math

import tensorflow.compat.v1 as tf

QUEUE_CAPACITY = 500
SHUFFLE_MIN_AFTER_DEQUEUE = QUEUE_CAPACITY // 5


def _shuffle_inputs(input_tensors, capacity, min_after_dequeue, num_threads):
  """Shuffles tensors in `input_tensors`, maintaining grouping."""
  shuffle_queue = tf.RandomShuffleQueue(
      capacity, min_after_dequeue, dtypes=[t.dtype for t in input_tensors])
  enqueue_op = shuffle_queue.enqueue(input_tensors)
  runner = tf.train.QueueRunner(shuffle_queue, [enqueue_op] * num_threads)
  tf.train.add_queue_runner(runner)

  output_tensors = shuffle_queue.dequeue()

  for i in range(len(input_tensors)):
    output_tensors[i].set_shape(input_tensors[i].shape)

  return output_tensors


def get_padded_batch(file_list, batch_size, input_size, label_shape=None,
                     num_enqueuing_threads=4, shuffle=False):
  """Reads batches of SequenceExamples from TFRecords and pads them.

  Can deal with variable length SequenceExamples by padding each batch to the
  length of the longest sequence with zeros.

  Args:
    file_list: A list of paths to TFRecord files containing SequenceExamples.
    batch_size: The number of SequenceExamples to include in each batch.
    input_size: The size of each input vector. The returned batch of inputs
        will have a shape [batch_size, num_steps, input_size].
    label_shape: Shape for labels. If not specified, will use [].
    num_enqueuing_threads: The number of threads to use for enqueuing
        SequenceExamples.
    shuffle: Whether to shuffle the batches.

  Returns:
    inputs: A tensor of shape [batch_size, num_steps, input_size] of floats32s.
    labels: A tensor of shape [batch_size, num_steps] of int64s.
    lengths: A tensor of shape [batch_size] of int32s. The lengths of each
        SequenceExample before padding.
  Raises:
    ValueError: If `shuffle` is True and `num_enqueuing_threads` is less than 2.
  """
  file_queue = tf.train.string_input_producer(file_list)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(file_queue)

  sequence_features = {
      'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                           dtype=tf.float32),
      'labels': tf.FixedLenSequenceFeature(shape=label_shape or [],
                                           dtype=tf.int64)}

  _, sequence = tf.parse_single_sequence_example(
      serialized_example, sequence_features=sequence_features)

  length = tf.shape(sequence['inputs'])[0]
  input_tensors = [sequence['inputs'], sequence['labels'], length]

  if shuffle:
    if num_enqueuing_threads < 2:
      raise ValueError(
          '`num_enqueuing_threads` must be at least 2 when shuffling.')
    shuffle_threads = int(math.ceil(num_enqueuing_threads) / 2.)

    # Since there may be fewer records than SHUFFLE_MIN_AFTER_DEQUEUE, take the
    # minimum of that number and the number of records.
    min_after_dequeue = count_records(
        file_list, stop_at=SHUFFLE_MIN_AFTER_DEQUEUE)
    input_tensors = _shuffle_inputs(
        input_tensors, capacity=QUEUE_CAPACITY,
        min_after_dequeue=min_after_dequeue,
        num_threads=shuffle_threads)

    num_enqueuing_threads -= shuffle_threads

  tf.logging.info(input_tensors)
  return tf.train.batch(
      input_tensors,
      batch_size=batch_size,
      capacity=QUEUE_CAPACITY,
      num_threads=num_enqueuing_threads,
      dynamic_pad=True,
      allow_smaller_final_batch=False)


def count_records(file_list, stop_at=None):
  """Counts number of records in files from `file_list` up to `stop_at`.

  Args:
    file_list: List of TFRecord files to count records in.
    stop_at: Optional number of records to stop counting at.

  Returns:
    Integer number of records in files from `file_list` up to `stop_at`.
  """
  num_records = 0
  for tfrecord_file in file_list:
    tf.logging.info('Counting records in %s.', tfrecord_file)
    for _ in tf.python_io.tf_record_iterator(tfrecord_file):
      num_records += 1
      if stop_at and num_records >= stop_at:
        tf.logging.info('Number of records is at least %d.', num_records)
        return num_records
  tf.logging.info('Total records: %d', num_records)
  return num_records


def flatten_maybe_padded_sequences(maybe_padded_sequences, lengths=None):
  """Flattens the batch of sequences, removing padding (if applicable).

  Args:
    maybe_padded_sequences: A tensor of possibly padded sequences to flatten,
        sized `[N, M, ...]` where M = max(lengths).
    lengths: Optional length of each sequence, sized `[N]`. If None, assumes no
        padding.

  Returns:
     flatten_maybe_padded_sequences: The flattened sequence tensor, sized
         `[sum(lengths), ...]`.
  """
  def flatten_unpadded_sequences():
    # The sequences are equal length, so we should just flatten over the first
    # two dimensions.
    return tf.reshape(maybe_padded_sequences,
                      [-1] + maybe_padded_sequences.shape.as_list()[2:])

  if lengths is None:
    return flatten_unpadded_sequences()

  def flatten_padded_sequences():
    indices = tf.where(tf.sequence_mask(lengths))
    return tf.gather_nd(maybe_padded_sequences, indices)

  return tf.cond(
      tf.equal(tf.reduce_min(lengths), tf.shape(maybe_padded_sequences)[1]),
      flatten_unpadded_sequences,
      flatten_padded_sequences)
