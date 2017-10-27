# Copyright 2016 Google Inc. All Rights Reserved.
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
# ==============================================================================
"""Small library that points to the ImageNet data set.

Methods of ImagenetData class:
  data_files: Returns a python list of all (sharded) data set files.
  num_examples_per_epoch: Returns the number of examples in the data set.
  num_classes: Returns the number of classes in the data set.
  reader: Return a reader for a single entry from the data set.

This file was taken nearly verbatim from the tensorflow/models GitHub repo.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('imagenet_data_dir', '/tmp/imagenet-2012-tfrecord',
                           """Path to the ImageNet data, i.e. """
                           """TFRecord of Example protos.""")


class ImagenetData(object):
  """A simple class for handling the ImageNet data set."""

  def __init__(self, subset):
    """Initialize dataset using a subset and the path to the data."""
    assert subset in self.available_subsets(), self.available_subsets()
    self.subset = subset

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 1000

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data set."""
    # Bounding box data consists of 615299 bounding boxes for 544546 images.
    if self.subset == 'train':
      return 1281167
    if self.subset == 'validation':
      return 50000

  def download_message(self):
    """Instruction to download and extract the tarball from Flowers website."""

    print('Failed to find any ImageNet %s files'% self.subset)
    print('')
    print('If you have already downloaded and processed the data, then make '
          'sure to set --imagenet_data_dir to point to the directory '
          'containing the location of the sharded TFRecords.\n')
    print('If you have not downloaded and prepared the ImageNet data in the '
          'TFRecord format, you will need to do this at least once. This '
          'process could take several hours depending on the speed of your '
          'computer and network connection\n')
    print('Please see '
          'https://github.com/tensorflow/models/blob/master/inception '
          'for instructions on how to build the ImageNet dataset using '
          'download_and_preprocess_imagenet.\n')
    print('Note that the raw data size is 300 GB and the processed data size '
          'is 150 GB. Please ensure you have at least 500GB disk space.')

  def available_subsets(self):
    """Returns the list of available subsets."""
    return ['train', 'validation']

  def data_files(self):
    """Returns a python list of all (sharded) data subset files.

    Returns:
      python list of all (sharded) data set files.

    Raises:
      ValueError: if there are not data_files matching the subset.
    """
    imagenet_data_dir = os.path.expanduser(FLAGS.imagenet_data_dir)
    if not tf.gfile.Exists(imagenet_data_dir):
      print('%s does not exist!' % (imagenet_data_dir))
      exit(-1)

    tf_record_pattern = os.path.join(imagenet_data_dir, '%s-*' % self.subset)
    data_files = tf.gfile.Glob(tf_record_pattern)
    if not data_files:
      print('No files found for dataset ImageNet/%s at %s' %
            (self.subset, imagenet_data_dir))

      self.download_message()
      exit(-1)

    return data_files

  def reader(self):
    """Return a reader for a single entry from the data set.

    See io_ops.py for details of Reader class.

    Returns:
      Reader object that reads the data set.
    """
    return tf.TFRecordReader()

