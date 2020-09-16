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

# pylint: skip-file
# TODO(adarob): Remove skip-file with https://github.com/PyCQA/astroid/issues/627
"""Train joint model on two latent spaces.

This script train the joint model defined in `model_joint.py` that transfers
between latent space of generative models that model the data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import importlib
import os

from magenta.models.latent_transfer import common
from magenta.models.latent_transfer import common_joint
from magenta.models.latent_transfer import model_joint
import numpy as np
import six
from six.moves import range
import tensorflow.compat.v1 as tf
from tqdm import tqdm

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('config', 'transfer_A_unconditional_mnist_to_mnist',
                       'The name of the model config to use.')
tf.flags.DEFINE_string('exp_uid_A', '_exp_0', 'exp_uid for data_A')
tf.flags.DEFINE_string('exp_uid_B', '_exp_1', 'exp_uid for data_B')
tf.flags.DEFINE_string('exp_uid', '_exp_0',
                       'String to append to config for filenames/directories.')
tf.flags.DEFINE_integer('n_iters', 100000, 'Number of iterations.')
tf.flags.DEFINE_integer('n_iters_per_save', 5000, 'Iterations per a save.')
tf.flags.DEFINE_integer('n_iters_per_eval', 5000,
                        'Iterations per a evaluation.')
tf.flags.DEFINE_integer('random_seed', 19260817, 'Random seed')
tf.flags.DEFINE_string('exp_uid_classifier', '_exp_0', 'exp_uid for classifier')

# For Overriding configs
tf.flags.DEFINE_integer('n_latent', 64, '')
tf.flags.DEFINE_integer('n_latent_shared', 2, '')
tf.flags.DEFINE_float('prior_loss_beta_A', 0.01, '')
tf.flags.DEFINE_float('prior_loss_beta_B', 0.01, '')
tf.flags.DEFINE_float('prior_loss_align_beta', 0.0, '')
tf.flags.DEFINE_float('mean_recons_A_align_beta', 0.0, '')
tf.flags.DEFINE_float('mean_recons_B_align_beta', 0.0, '')
tf.flags.DEFINE_float('mean_recons_A_to_B_align_beta', 0.0, '')
tf.flags.DEFINE_float('mean_recons_B_to_A_align_beta', 0.0, '')
tf.flags.DEFINE_integer('pairing_number', 1024, '')


def load_config(config_name):
  return importlib.import_module('configs.%s' % config_name).config


def main(unused_argv):
  # pylint:disable=unused-variable
  # Reason:
  #   This training script relys on many programmatical call to function and
  #   access to variables. Pylint cannot infer this case so it emits false alarm
  #   of unused-variable if we do not disable this warning.

  # pylint:disable=invalid-name
  # Reason:
  #   Following variables have their name consider to be invalid by pylint so
  #   we disable the warning.
  #   - Variable that in its name has A or B indicating their belonging of
  #     one side of data.
  del unused_argv

  # Load main config
  config_name = FLAGS.config
  config = load_config(config_name)

  config_name_A = config['config_A']
  config_name_B = config['config_B']
  config_name_classifier_A = config['config_classifier_A']
  config_name_classifier_B = config['config_classifier_B']

  # Load dataset
  dataset_A = common_joint.load_dataset(config_name_A, FLAGS.exp_uid_A)
  (dataset_blob_A, train_data_A, train_label_A, train_mu_A, train_sigma_A,
   index_grouped_by_label_A) = dataset_A
  dataset_B = common_joint.load_dataset(config_name_B, FLAGS.exp_uid_B)
  (dataset_blob_B, train_data_B, train_label_B, train_mu_B, train_sigma_B,
   index_grouped_by_label_B) = dataset_B

  # Prepare directories
  dirs = common_joint.prepare_dirs('joint', config_name, FLAGS.exp_uid)
  save_dir, sample_dir = dirs

  # Set random seed
  np.random.seed(FLAGS.random_seed)
  tf.set_random_seed(FLAGS.random_seed)

  # Real Training.
  tf.reset_default_graph()
  sess = tf.Session()

  # Load model's architecture (= build)
  one_side_helper_A = common_joint.OneSideHelper(config_name_A, FLAGS.exp_uid_A,
                                                 config_name_classifier_A,
                                                 FLAGS.exp_uid_classifier)
  one_side_helper_B = common_joint.OneSideHelper(config_name_B, FLAGS.exp_uid_B,
                                                 config_name_classifier_B,
                                                 FLAGS.exp_uid_classifier)
  m = common_joint.load_model(model_joint.Model, config_name, FLAGS.exp_uid)

  # Prepare summary
  train_writer = tf.summary.FileWriter(save_dir + '/transfer_train', sess.graph)
  scalar_summaries = tf.summary.merge([
      tf.summary.scalar(key, value)
      for key, value in m.get_summary_kv_dict().items()
  ])
  manual_summary_helper = common_joint.ManualSummaryHelper()

  # Initialize and restore
  sess.run(tf.global_variables_initializer())

  one_side_helper_A.restore(dataset_blob_A)
  one_side_helper_B.restore(dataset_blob_B)

  # Miscs from config
  batch_size = config['batch_size']
  n_latent_shared = config['n_latent_shared']
  pairing_number = config['pairing_number']
  n_latent_A = config['vae_A']['n_latent']
  n_latent_B = config['vae_B']['n_latent']
  i_start = 0
  # Data iterators

  single_data_iterator_A = common_joint.SingleDataIterator(
      train_mu_A, train_sigma_A, batch_size)
  single_data_iterator_B = common_joint.SingleDataIterator(
      train_mu_B, train_sigma_B, batch_size)
  paired_data_iterator = common_joint.PairedDataIterator(
      train_mu_A, train_sigma_A, train_data_A, train_label_A,
      index_grouped_by_label_A, train_mu_B, train_sigma_B, train_data_B,
      train_label_B, index_grouped_by_label_B, pairing_number, batch_size)
  single_data_iterator_A_for_evaluation = common_joint.SingleDataIterator(
      train_mu_A, train_sigma_A, batch_size)
  single_data_iterator_B_for_evaluation = common_joint.SingleDataIterator(
      train_mu_B, train_sigma_B, batch_size)

  # Training loop
  n_iters = FLAGS.n_iters
  for i in tqdm(list(range(i_start, n_iters)), desc='training', unit=' batch'):
    # Prepare data for this batch
    # - Unsupervised (A)
    x_A, _ = next(single_data_iterator_A)
    x_B, _ = next(single_data_iterator_B)
    # - Supervised (aligning)
    x_align_A, x_align_B, align_debug_info = next(paired_data_iterator)
    real_x_align_A, real_x_align_B = align_debug_info

    # Run training op and write summary
    res = sess.run([m.train_full, scalar_summaries], {
        m.x_A: x_A,
        m.x_B: x_B,
        m.x_align_A: x_align_A,
        m.x_align_B: x_align_B,
    })
    train_writer.add_summary(res[-1], i)

    if i % FLAGS.n_iters_per_save == 0:
      # Save the model if instructed
      config_name = FLAGS.config
      model_uid = common.get_model_uid(config_name, FLAGS.exp_uid)

      save_name = os.path.join(save_dir, 'transfer_%s_%d.ckpt' % (model_uid, i))
      m.vae_saver.save(sess, save_name)
      with tf.gfile.Open(os.path.join(save_dir, 'ckpt_iters.txt'), 'w') as f:
        f.write('%d' % i)

    # Evaluate if instructed
    if i % FLAGS.n_iters_per_eval == 0:
      # Helper functions
      def joint_sample(sample_size):
        z_hat = np.random.randn(sample_size, n_latent_shared)
        return sess.run([m.x_joint_A, m.x_joint_B], {
            m.z_hat: z_hat,
        })

      def get_x_from_prior_A():
        return sess.run(m.x_from_prior_A)

      def get_x_from_prior_B():
        return sess.run(m.x_from_prior_B)

      def get_x_from_posterior_A():
        return next(single_data_iterator_A_for_evaluation)[0]

      def get_x_from_posterior_B():
        return next(single_data_iterator_B_for_evaluation)[0]

      def get_x_prime_A(x_A):
        return sess.run(m.x_prime_A, {m.x_A: x_A})

      def get_x_prime_B(x_B):
        return sess.run(m.x_prime_B, {m.x_B: x_B})

      def transfer_A_to_B(x_A):
        return sess.run(m.x_A_to_B, {m.x_A: x_A})

      def transfer_B_to_A(x_B):
        return sess.run(m.x_B_to_A, {m.x_B: x_B})

      def manual_summary(key, value):
        summary = manual_summary_helper.get_summary(sess, key, value)
        # This [cell-var-from-loop] is intented
        train_writer.add_summary(summary, i)  # pylint: disable=cell-var-from-loop

      # Classifier based evaluation
      sample_total_size = 10000
      sample_batch_size = 100

      def pred(one_side_helper, x):
        real_x = six.ensure_text(one_side_helper.m_helper, x)
        return one_side_helper.m_classifier_helper.classify(real_x, batch_size)

      def accuarcy(x_1, x_2, type_1, type_2):
        assert type_1 in ('A', 'B') and type_2 in ('A', 'B')
        func_A = partial(pred, one_side_helper=one_side_helper_A)
        func_B = partial(pred, one_side_helper=one_side_helper_B)
        func_1 = func_A if type_1 == 'A' else func_B
        func_2 = func_A if type_2 == 'A' else func_B
        pred_1, pred_2 = func_1(x=x_1), func_2(x=x_2)
        return np.mean(np.equal(pred_1, pred_2).astype('f'))

      def joint_sample_accuarcy():
        x_A, x_B = joint_sample(sample_size=sample_total_size)  # pylint: disable=cell-var-from-loop
        return accuarcy(x_A, x_B, 'A', 'B')

      def transfer_sample_accuarcy_A_B():
        x_A = get_x_from_prior_A()
        x_B = transfer_A_to_B(x_A)
        return accuarcy(x_A, x_B, 'A', 'B')

      def transfer_sample_accuarcy_B_A():
        x_B = get_x_from_prior_B()
        x_A = transfer_B_to_A(x_B)
        return accuarcy(x_A, x_B, 'A', 'B')

      def transfer_accuarcy_A_B():
        x_A = get_x_from_posterior_A()
        x_B = transfer_A_to_B(x_A)
        return accuarcy(x_A, x_B, 'A', 'B')

      def transfer_accuarcy_B_A():
        x_B = get_x_from_posterior_B()
        x_A = transfer_B_to_A(x_B)
        return accuarcy(x_A, x_B, 'A', 'B')

      def recons_accuarcy_A():
        # Use x_A in outer scope
        # These [cell-var-from-loop]s are intended
        x_A_prime = get_x_prime_A(x_A)  # pylint: disable=cell-var-from-loop
        return accuarcy(x_A, x_A_prime, 'A', 'A')  # pylint: disable=cell-var-from-loop

      def recons_accuarcy_B():
        # use x_B in outer scope
        # These [cell-var-from-loop]s are intended
        x_B_prime = get_x_prime_B(x_B)  # pylint: disable=cell-var-from-loop
        return accuarcy(x_B, x_B_prime, 'B', 'B')  # pylint: disable=cell-var-from-loop

      # Do all manual summary
      for func_name in (
          'joint_sample_accuarcy',
          'transfer_sample_accuarcy_A_B',
          'transfer_sample_accuarcy_B_A',
          'transfer_accuarcy_A_B',
          'transfer_accuarcy_B_A',
          'recons_accuarcy_A',
          'recons_accuarcy_B',
      ):
        func = locals()[func_name]
        manual_summary(func_name, func())

      # Sampling based evaluation / sampling
      x_prime_A = get_x_prime_A(x_A)
      x_prime_B = get_x_prime_B(x_B)
      x_from_prior_A = get_x_from_prior_A()
      x_from_prior_B = get_x_from_prior_B()
      x_A_to_B = transfer_A_to_B(x_A)
      x_B_to_A = transfer_B_to_A(x_B)
      x_align_A_to_B = transfer_A_to_B(x_align_A)
      x_align_B_to_A = transfer_B_to_A(x_align_B)
      x_joint_A, x_joint_B = joint_sample(sample_size=batch_size)

      this_iter_sample_dir = os.path.join(
          sample_dir, 'transfer_train_sample', '%010d' % i)
      tf.gfile.MakeDirs(this_iter_sample_dir)

      for helper, var_names, x_is_real_x in [
          (one_side_helper_A.m_helper,
           ('x_A', 'x_prime_A', 'x_from_prior_A', 'x_B_to_A', 'x_align_A',
            'x_align_B_to_A', 'x_joint_A'), False),
          (one_side_helper_A.m_helper, ('real_x_align_A',), True),
          (one_side_helper_B.m_helper,
           ('x_B', 'x_prime_B', 'x_from_prior_B', 'x_A_to_B', 'x_align_B',
            'x_align_A_to_B', 'x_joint_B'), False),
          (one_side_helper_B.m_helper, ('real_x_align_B',), True),
      ]:
        for var_name in var_names:
          # Here `var` would be None if
          #   - there is no such variable in `locals()`, or
          #   - such variable exists but the value is None
          # In both case, we would skip saving data from it.
          var = locals().get(var_name, None)
          if var is not None:
            helper.save_data(var, var_name, this_iter_sample_dir, x_is_real_x)

  # pylint:enable=invalid-name
  # pylint:enable=unused-variable


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.app.run(main)
