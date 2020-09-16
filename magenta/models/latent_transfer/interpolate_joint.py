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

"""Produce interpolation in the joint model trained by `train_joint.py`.

This script produces interpolation on one side of the joint model as a series of
images, as well as in other side of the model through paralleling,
image-by-image transformation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os

from magenta.models.latent_transfer import common
from magenta.models.latent_transfer import common_joint
from magenta.models.latent_transfer import model_joint
import numpy as np
import tensorflow.compat.v1 as tf

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

# For controling interpolation
tf.flags.DEFINE_integer('load_ckpt_iter', 0, '')
tf.flags.DEFINE_string('interpolate_labels', '',
                       'a `,` separated list of 0-indexed labels.')
tf.flags.DEFINE_integer('nb_images_between_labels', 1, '')


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

  # Initialize and restore
  sess.run(tf.global_variables_initializer())

  one_side_helper_A.restore(dataset_blob_A)
  one_side_helper_B.restore(dataset_blob_B)

  # Restore from ckpt
  config_name = FLAGS.config
  model_uid = common.get_model_uid(config_name, FLAGS.exp_uid)
  save_name = os.path.join(
      save_dir, 'transfer_%s_%d.ckpt' % (model_uid, FLAGS.load_ckpt_iter))
  m.vae_saver.restore(sess, save_name)

  # prepare intepolate dir
  intepolate_dir = os.path.join(
      sample_dir, 'interpolate_sample', '%010d' % FLAGS.load_ckpt_iter)
  tf.gfile.MakeDirs(intepolate_dir)

  # things
  interpolate_labels = [int(_) for _ in FLAGS.interpolate_labels.split(',')]
  nb_images_between_labels = FLAGS.nb_images_between_labels

  index_list_A = []
  last_pos = [0] * 10
  for label in interpolate_labels:
    index_list_A.append(index_grouped_by_label_A[label][last_pos[label]])
    last_pos[label] += 1

  index_list_B = []
  last_pos = [-1] * 10
  for label in interpolate_labels:
    index_list_B.append(index_grouped_by_label_B[label][last_pos[label]])
    last_pos[label] -= 1

  z_A = []
  z_A.append(train_mu_A[index_list_A[0]])
  for i_label in range(1, len(interpolate_labels)):
    last_z_A = z_A[-1]
    this_z_A = train_mu_A[index_list_A[i_label]]
    for j in range(1, nb_images_between_labels + 1):
      z_A.append(last_z_A +
                 (this_z_A - last_z_A) * (float(j) / nb_images_between_labels))
  z_B = []
  z_B.append(train_mu_B[index_list_B[0]])
  for i_label in range(1, len(interpolate_labels)):
    last_z_B = z_B[-1]
    this_z_B = train_mu_B[index_list_B[i_label]]
    for j in range(1, nb_images_between_labels + 1):
      z_B.append(last_z_B +
                 (this_z_B - last_z_B) * (float(j) / nb_images_between_labels))
  z_B_tr = []
  for this_z_A in z_A:
    this_z_B_tr = sess.run(m.x_A_to_B_direct, {m.x_A: np.array([this_z_A])})
    z_B_tr.append(this_z_B_tr[0])

  # Generate data domain instances and save.
  z_A = np.array(z_A)
  x_A = one_side_helper_A.m_helper.decode(z_A)
  x_A = common.post_proc(x_A, one_side_helper_A.m_helper.config)
  batched_x_A = common.batch_image(
      x_A,
      max_images=len(x_A),
      rows=len(x_A),
      cols=1,
  )
  common.save_image(batched_x_A, os.path.join(intepolate_dir, 'x_A.png'))

  z_B = np.array(z_B)
  x_B = one_side_helper_B.m_helper.decode(z_B)
  x_B = common.post_proc(x_B, one_side_helper_B.m_helper.config)
  batched_x_B = common.batch_image(
      x_B,
      max_images=len(x_B),
      rows=len(x_B),
      cols=1,
  )
  common.save_image(batched_x_B, os.path.join(intepolate_dir, 'x_B.png'))

  z_B_tr = np.array(z_B_tr)
  x_B_tr = one_side_helper_B.m_helper.decode(z_B_tr)
  x_B_tr = common.post_proc(x_B_tr, one_side_helper_B.m_helper.config)
  batched_x_B_tr = common.batch_image(
      x_B_tr,
      max_images=len(x_B_tr),
      rows=len(x_B_tr),
      cols=1,
  )
  common.save_image(batched_x_B_tr, os.path.join(intepolate_dir, 'x_B_tr.png'))


if __name__ == '__main__':
  tf.app.run(main)
