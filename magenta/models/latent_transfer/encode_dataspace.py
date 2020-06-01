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
"""Encode the data using pre-trained VAE on dataspace.

This script encodes the instances in dataspace (x) from the training set into
distributions in the latent space (z) using the pre-trained the models from
`train_dataspace.py`
"""

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


def main(unused_argv):
  del unused_argv

  # Load Config
  config_name = FLAGS.config
  config_module = importlib.import_module('configs.%s' % config_name)
  config = config_module.config
  model_uid = common.get_model_uid(config_name, FLAGS.exp_uid)
  batch_size = config['batch_size']

  # Load dataset
  dataset = common.load_dataset(config)
  basepath = dataset.basepath
  save_path = dataset.save_path
  train_data = dataset.train_data
  eval_data = dataset.eval_data

  # Make the directory
  save_dir = os.path.join(save_path, model_uid)
  best_dir = os.path.join(save_dir, 'best')
  tf.gfile.MakeDirs(save_dir)
  tf.gfile.MakeDirs(best_dir)
  tf.logging.info('Save Dir: %s', save_dir)

  # Load Model
  tf.reset_default_graph()
  sess = tf.Session()
  m = model_dataspace.Model(config, name=model_uid)
  _ = m()  # noqa

  # Initialize
  sess.run(tf.global_variables_initializer())

  # Load
  m.vae_saver.restore(sess,
                      os.path.join(best_dir, 'vae_best_%s.ckpt' % model_uid))

  # Encode
  def encode(data):
    """Encode the data in dataspace to latent spaceself.

    This script runs the encoding in batched mode to limit GPU memory usage.

    Args:
      data: A numpy array of data to be encoded.

    Returns:
      A object with instances `mu` and `sigma`, the parameters of encoded
      distributions in the latent space.
    """
    mu_list, sigma_list = [], []

    for i in range(0, len(data), batch_size):
      start, end = i, min(i + batch_size, len(data))
      batch = data[start:end]

      mu, sigma = sess.run([m.mu, m.sigma], {m.x: batch})
      mu_list.append(mu)
      sigma_list.append(sigma)

    mu = np.concatenate(mu_list)
    sigma = np.concatenate(sigma_list)

    return common.ObjectBlob(mu=mu, sigma=sigma)

  encoded_train_data = encode(train_data)
  tf.logging.info(
      'encode train_data: mu.shape = %s sigma.shape = %s',
      encoded_train_data.mu.shape,
      encoded_train_data.sigma.shape,
  )

  encoded_eval_data = encode(eval_data)
  tf.logging.info(
      'encode eval_data: mu.shape = %s sigma.shape = %s',
      encoded_eval_data.mu.shape,
      encoded_eval_data.sigma.shape,
  )

  # Save encoded as npz file
  encoded_save_path = os.path.join(basepath, 'encoded', model_uid)
  tf.gfile.MakeDirs(encoded_save_path)
  tf.logging.info('encoded train_data saved to %s',
                  os.path.join(encoded_save_path, 'encoded_train_data.npz'))
  np.savez(
      os.path.join(encoded_save_path, 'encoded_train_data.npz'),
      mu=encoded_train_data.mu,
      sigma=encoded_train_data.sigma,
  )
  tf.logging.info('encoded eval_data saved to %s',
                  os.path.join(encoded_save_path, 'encoded_eval_data.npz'))
  np.savez(
      os.path.join(encoded_save_path, 'encoded_eval_data.npz'),
      mu=encoded_eval_data.mu,
      sigma=encoded_eval_data.sigma,
  )


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.app.run(main)
