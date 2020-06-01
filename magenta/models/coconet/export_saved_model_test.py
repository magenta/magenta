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

"""Tests for export_saved_model."""
import os
import tempfile

from magenta.models.coconet import export_saved_model
from magenta.models.coconet import lib_graph
from magenta.models.coconet import lib_hparams
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class ExportSavedModelTest(tf.test.TestCase):

  def save_checkpoint(self):
    logdir = tempfile.mkdtemp()
    save_path = os.path.join(logdir, 'model.ckpt')

    hparams = lib_hparams.Hyperparameters(**{})

    tf.gfile.MakeDirs(logdir)
    config_fpath = os.path.join(logdir, 'config')
    with tf.gfile.Open(config_fpath, 'w') as p:
      hparams.dump(p)

    with tf.Graph().as_default():
      lib_graph.build_graph(is_training=True, hparams=hparams)
      sess = tf.Session()
      sess.run(tf.global_variables_initializer())

      saver = tf.train.Saver()
      saver.save(sess, save_path)

    return logdir

  def test_export_saved_model(self):
    checkpoint_path = self.save_checkpoint()

    destination_dir = os.path.join(checkpoint_path, 'export')

    export_saved_model.export(checkpoint_path, destination_dir,
                              use_tf_sampling=True)

    self.assertTrue(tf.gfile.Exists(
        os.path.join(destination_dir, 'saved_model.pb')))
    tf.gfile.DeleteRecursively(checkpoint_path)


if __name__ == '__main__':
  tf.test.main()
