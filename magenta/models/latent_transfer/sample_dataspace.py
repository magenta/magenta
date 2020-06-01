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

"""Sample from pre-trained VAE on dataspace.

This script provides sampling from VAE on dataspace trained using
`train_dataspace.py`. The main purpose is to help manually check the quality
of model on dataspace.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os

from magenta.models.latent_transfer import common
from magenta.models.latent_transfer import model_dataspace
import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('config', 'mnist_0',
                       'The name of the model config to use.')
tf.flags.DEFINE_string('exp_uid', '_exp_0',
                       'String to append to config for filenames/directories.')
tf.flags.DEFINE_integer('random_seed', 19260817, 'Random seed.')


def main(unused_argv):
  del unused_argv

  # Load Config
  config_name = FLAGS.config
  config_module = importlib.import_module('configs.%s' % config_name)
  config = config_module.config
  model_uid = common.get_model_uid(config_name, FLAGS.exp_uid)
  n_latent = config['n_latent']

  # Load dataset
  dataset = common.load_dataset(config)
  basepath = dataset.basepath
  save_path = dataset.save_path
  train_data = dataset.train_data

  # Make the directory
  save_dir = os.path.join(save_path, model_uid)
  best_dir = os.path.join(save_dir, 'best')
  tf.gfile.MakeDirs(save_dir)
  tf.gfile.MakeDirs(best_dir)
  tf.logging.info('Save Dir: %s', save_dir)

  # Set random seed
  np.random.seed(FLAGS.random_seed)
  tf.set_random_seed(FLAGS.random_seed)

  # Load Model
  tf.reset_default_graph()
  sess = tf.Session()
  with tf.device(tf.train.replica_device_setter(ps_tasks=0)):
    m = model_dataspace.Model(config, name=model_uid)
    _ = m()  # noqa

    # Initialize
    sess.run(tf.global_variables_initializer())

    # Load
    m.vae_saver.restore(sess,
                        os.path.join(best_dir, 'vae_best_%s.ckpt' % model_uid))

    # Sample from prior
    sample_count = 64

    image_path = os.path.join(basepath, 'sample', model_uid)
    tf.gfile.MakeDirs(image_path)

    # from prior
    z_p = np.random.randn(sample_count, m.n_latent)
    x_p = sess.run(m.x_mean, {m.z: z_p})
    x_p = common.post_proc(x_p, config)
    common.save_image(
        common.batch_image(x_p), os.path.join(image_path, 'sample_prior.png'))

    # Sample from priro, as Grid
    boundary = 2.0
    number_grid = 50
    blob = common.make_grid(
        boundary=boundary, number_grid=number_grid, dim_latent=n_latent)
    z_grid, dim_grid = blob.z_grid, blob.dim_grid
    x_grid = sess.run(m.x_mean, {m.z: z_grid})
    x_grid = common.post_proc(x_grid, config)
    batch_image_grid = common.make_batch_image_grid(dim_grid, number_grid)
    common.save_image(
        batch_image_grid(x_grid), os.path.join(image_path, 'sample_grid.png'))

    # Reconstruction
    sample_count = 64
    x_real = train_data[:sample_count]
    mu, sigma = sess.run([m.mu, m.sigma], {m.x: x_real})
    x_rec = sess.run(m.x_mean, {m.mu: mu, m.sigma: sigma})
    x_rec = common.post_proc(x_rec, config)

    x_real = common.post_proc(x_real, config)
    common.save_image(
        common.batch_image(x_real), os.path.join(image_path, 'image_real.png'))
    common.save_image(
        common.batch_image(x_rec), os.path.join(image_path, 'image_rec.png'))


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.app.run(main)
