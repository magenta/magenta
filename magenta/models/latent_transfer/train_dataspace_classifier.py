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

# Lint as: python3
"""Train classifier on dataspace.

This script trains the data space classifier as defined in
`model_dataspace.py`. The best checkpoint (as evaluated on valid set)
would be used to classifier instances in the data space (x).
"""

# pylint:disable=invalid-name

import importlib
import os

from magenta.models.latent_transfer import common
from magenta.models.latent_transfer import model_dataspace
import numpy as np
import tensorflow.compat.v1 as tf

configs_module_prefix = 'magenta.models.latent_transfer.configs'

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('config', 'mnist_0',
                       'The name of the model config to use.')
tf.flags.DEFINE_bool('local', False, 'Run job locally.')
tf.flags.DEFINE_integer('n_iters', 200000, 'Number of iterations.')
tf.flags.DEFINE_integer('n_iters_per_save', 10000, 'Iterations per a save.')
tf.flags.DEFINE_float('lr', 3e-4, 'learning_rate.')
tf.flags.DEFINE_string('exp_uid', '_exp_0',
                       'String to append to config for filenames/directories.')


def main(unused_argv):
  del unused_argv

  # Load Config
  config_name = FLAGS.config
  config_module = importlib.import_module(configs_module_prefix +
                                          '.%s' % config_name)
  config = config_module.config
  model_uid = common.get_model_uid(config_name, FLAGS.exp_uid)
  batch_size = config['batch_size']

  # Load dataset
  dataset = common.load_dataset(config)
  save_path = dataset.save_path
  train_data = dataset.train_data
  attr_train = dataset.attr_train
  eval_data = dataset.eval_data
  attr_eval = dataset.attr_eval

  # Make the directory
  save_dir = os.path.join(save_path, model_uid)
  best_dir = os.path.join(save_dir, 'best')
  tf.gfile.MakeDirs(save_dir)
  tf.gfile.MakeDirs(best_dir)
  tf.logging.info('Save Dir: %s', save_dir)

  np.random.seed(10003)
  N_train = train_data.shape[0]
  N_eval = eval_data.shape[0]

  # Load Model
  tf.reset_default_graph()
  sess = tf.Session()
  m = model_dataspace.Model(config, name=model_uid)
  _ = m()  # noqa

  # Create summaries
  y_true = m.labels
  y_pred = tf.cast(tf.greater(m.pred_classifier, 0.5), tf.int32)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

  tf.summary.scalar('Loss', m.classifier_loss)
  tf.summary.scalar('Accuracy', accuracy)
  scalar_summaries = tf.summary.merge_all()

  # Summary writers
  train_writer = tf.summary.FileWriter(save_dir + '/train', sess.graph)
  eval_writer = tf.summary.FileWriter(save_dir + '/eval', sess.graph)

  # Initialize
  sess.run(tf.global_variables_initializer())

  i_start = 0
  running_N_eval = 30
  traces = {
      'i': [],
      'i_pred': [],
      'loss': [],
      'loss_eval': [],
  }

  best_eval_loss = np.inf
  classifier_lr_ = np.logspace(
      np.log10(FLAGS.lr), np.log10(1e-6), FLAGS.n_iters)

  # Train the Classifier
  for i in range(i_start, FLAGS.n_iters):
    start = (i * batch_size) % N_train
    end = start + batch_size
    batch = train_data[start:end]
    labels = attr_train[start:end]

    # train op
    res = sess.run([m.train_classifier, m.classifier_loss, scalar_summaries], {
        m.x: batch,
        m.labels: labels,
        m.classifier_lr: classifier_lr_[i]
    })
    tf.logging.info('Iter: %d, Loss: %.2e', i, res[1])
    train_writer.add_summary(res[-1], i)

    if i % 10 == 0:
      # write training reconstructions
      if batch.shape[0] == batch_size:
        # write eval summaries
        start = (i * batch_size) % N_eval
        end = start + batch_size
        batch = eval_data[start:end]
        labels = attr_eval[start:end]

        if batch.shape[0] == batch_size:
          res_eval = sess.run([m.classifier_loss, scalar_summaries], {
              m.x: batch,
              m.labels: labels,
          })
          traces['loss_eval'].append(res_eval[0])
          eval_writer.add_summary(res_eval[-1], i)

    if i % FLAGS.n_iters_per_save == 0:
      smoothed_eval_loss = np.mean(traces['loss_eval'][-running_N_eval:])
      if smoothed_eval_loss < best_eval_loss:

        # Save the best model
        best_eval_loss = smoothed_eval_loss
        save_name = os.path.join(best_dir,
                                 'classifier_best_%s.ckpt' % model_uid)
        tf.logging.info('SAVING BEST! %s Iter: %d', save_name, i)
        m.classifier_saver.save(sess, save_name)
        with tf.gfile.Open(os.path.join(best_dir, 'best_ckpt_iters.txt'),
                           'w') as f:
          f.write('%d' % i)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.app.run(main)
