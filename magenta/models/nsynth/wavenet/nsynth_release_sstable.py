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
"""Module to load the Dataset."""

# internal imports
import tensorflow as tf


class NSynthDataset(object):
  """Dataset object to help manage the SSTable loading."""

  def __init__(self, train_path, test_path, subset="train", is_training=True,
               length=64000):
    self.length = length
    self.is_training = is_training
    self.num_examples = self._num_examples(subset)
    self.expression_paths = self._paths(train_path, test_path)[subset]

    ending = train_path.split("@")[0]
    if ending.endswith(".ss") or ending.endswith(".sstable"):
      self.reader_class = tf.SSTableReader
    elif ending.endswith(".tfrecord"):
      self.reader_class = tf.TFRecordReader

  @staticmethod
  def _num_examples(subset):
    return {
        "train": 274917,
        "train_dev": 14352,
        "valid_dev": 2040,
        "valid_test": 2056
    }.get(subset)

  @staticmethod
  def _paths(train_path, test_path):
    """Get the file paths for the Queues.

    Args:
      train_path: The path to the training shards.
      test_path: The path to the test shards.

    Returns:
      A dict of key to file paths, where the keys correspond to the data split.
      They are "train", "train_dev", "valid_dev", and "valid_test".
    """
    path, num_shards = train_path.split("@")
    num_total_shards = int(num_shards)
    num_train_shards = int(round((1 - 0.05) * num_total_shards))
    path_list_train = [
        "%s-%05d-of-%05d" % (path, i, num_total_shards)
        for i in xrange(0, num_train_shards)
    ]
    path_list_train_dev = [
        "%s-%05d-of-%05d" % (path, i, num_total_shards)
        for i in xrange(num_train_shards, num_total_shards)
    ]

    path, num_shards = test_path.split("@")
    num_total_shards = int(num_shards)
    num_test_shards = int(round((.5 * num_total_shards)))
    path_list_valid_dev = [
        "%s-%05d-of-%05d" % (path, i, num_total_shards)
        for i in xrange(0, num_test_shards)
    ]
    path_list_valid_test = [
        "%s-%05d-of-%05d" % (path, i, num_total_shards)
        for i in xrange(num_test_shards, num_total_shards)
    ]

    return {
        "train": path_list_train,
        "train_dev": path_list_train_dev,
        "valid_dev": path_list_valid_dev,
        "valid_test": path_list_valid_test
    }

  def get_expressions(self, batch_size):
    """Get the Tensor expressions from the reader.

    Args:
      batch_size: The integer batch size.

    Returns:
      A dict of key:tensor pairs. This includes "pitch", "wav", and "key".
    """
    reader = self._reader_class()

    num_epochs = None if self.is_training else 1
    capacity = batch_size
    path_queue = tf.input_producer(
        self.expression_paths,
        num_epochs=num_epochs,
        shuffle=self.is_training,
        capacity=capacity)
    key, serialized_example = reader.read(path_queue)

    features = {
        "pitch": tf.FixedLenFeature([1], dtype=tf.int64),
        "audio": tf.FixedLenFeature([64000], dtype=tf.float32),
        "qualities": tf.FixedLenFeature([10], dtype=tf.int64),
        "instrument_source": tf.FixedLenFeature([1], dtype=tf.int64),
        "instrument_family": tf.FixedLenFeature([1], dtype=tf.int64),
    }

    example = tf.parse_single_example(serialized_example, features)
    wav = example["audio"]
    wav = tf.slice(wav, [0], [64000])
    pitch = tf.squeeze(example["pitch"])

    if self.is_training:
      # random crop
      crop = tf.random_crop(wav, [self.length])
      crop = tf.reshape(crop, [1, self.length])
      key, crop, pitch = tf.shuffle_batch(
          [key, crop, pitch],
          batch_size,
          num_threads=4,
          capacity=500 * batch_size,
          min_after_dequeue=200 * batch_size)
    else:
      # fixed center crop
      offset = (64000 - self.length) // 2  # 24320
      crop = tf.slice(wav, [offset], [self.length])
      crop = tf.reshape(crop, [1, self.length])
      key, crop, pitch = tf.shuffle_batch(
          [key, crop, pitch],
          batch_size,
          num_threads=4,
          capacity=500 * batch_size,
          min_after_dequeue=200 * batch_size)

    crop = tf.reshape(tf.cast(crop, tf.float32), [batch_size, self.length])
    pitch = tf.cast(pitch, tf.int32)
    return {"pitch": pitch, "wav": crop, "key": key}
