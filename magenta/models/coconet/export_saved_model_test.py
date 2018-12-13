"""Tests for export_saved_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import tensorflow as tf

from magenta.models.coconet import export_saved_model
from magenta.models.coconet import lib_graph
from magenta.models.coconet import lib_hparams


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
