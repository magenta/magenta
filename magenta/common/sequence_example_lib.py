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
"""Utility functions for working with tf.train.SequenceExamples."""

import tensorflow as tf


def make_sequence_example(inputs, labels):
  """Returns a SequenceExample for the given inputs and labels.

  Args:
    inputs: A list of input vectors. Each input vector is a list of floats.
    labels: A list of ints.

  Returns:
    A tf.train.SequenceExample containing inputs and labels.
  """
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


def get_padded_batch(file_list, batch_size, input_size,
                     num_enqueuing_threads=4):
  """Reads batches of SequenceExamples from TFRecords and pads them.

  Can deal with variable length SequenceExamples by padding each batch to the
  length of the longest sequence with zeros.

  Args:
    file_list: A list of paths to TFRecord files containing SequenceExamples.
    batch_size: The number of SequenceExamples to include in each batch.
    input_size: The size of each input vector. The returned batch of inputs
        will have a shape [batch_size, num_steps, input_size].
    num_enqueuing_threads: The number of threads to use for enqueuing
        SequenceExamples.

  Returns:
    inputs: A tensor of shape [batch_size, num_steps, input_size] of floats32s.
    labels: A tensor of shape [batch_size, num_steps] of int64s.
    lengths: A tensor of shape [batch_size] of int32s. The lengths of each
        SequenceExample before padding.
  """
  file_queue = tf.train.string_input_producer(file_list)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(file_queue)

  sequence_features = {
      'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                           dtype=tf.float32),
      'labels': tf.FixedLenSequenceFeature(shape=[],
                                           dtype=tf.int64)}

  _, sequence = tf.parse_single_sequence_example(
      serialized_example, sequence_features=sequence_features)

  length = tf.shape(sequence['inputs'])[0]

  queue = tf.PaddingFIFOQueue(
      capacity=1000,
      dtypes=[tf.float32, tf.int64, tf.int32],
      shapes=[(None, input_size), (None,), ()])

  enqueue_ops = [queue.enqueue([sequence['inputs'],
                                sequence['labels'],
                                length])] * num_enqueuing_threads
  tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))
  return queue.dequeue_many(batch_size)
