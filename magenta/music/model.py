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
"""Abstract class for models.

Provides a uniform interface for interacting with any model.
"""

import abc

# internal imports

import tensorflow as tf


class BaseModel(object):
  """Abstract class for models.

  Implements default session checkpoint restore methods.
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self):
    """Constructs a BaseModel."""
    self._session = None

  @abc.abstractmethod
  def _build_graph_for_generation(self):
    """Builds and returns the model graph for generation.

    Will be called before restoring a checkpoint file.

    Returns:
      The tf.Graph object.
    """
    pass

  def initialize_with_checkpoint(self, checkpoint_file):
    graph = self._build_graph()
    with graph.as_default():
      saver = tf.train.Saver()
      self._session = tf.Session()
      tf.logging.info('Checkpoint used: %s', checkpoint_file)
      saver.restore(self._session, checkpoint_file)

  def initialize_with_checkpoint_and_metagraph(self, checkpoint_filename,
                                               metagraph_filename):
    with tf.Graph().as_default():
      self._session = tf.Session()
      new_saver = tf.train.import_meta_graph(metagraph_filename)
      new_saver.restore(self._session, checkpoint_filename)

  def write_checkpoint_with_metagraph(self, checkpoint_filename):
    with self._session.graph.as_default():
      saver = tf.train.Saver(sharded=False)
      saver.save(self._session, checkpoint_filename, meta_graph_suffix='meta',
                 write_meta_graph=True)

  def close(self):
    self._session.close()
    self._session = None

