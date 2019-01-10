# Copyright 2019 Google Inc. All Rights Reserved.
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
"""Code for computing neural net divergences, uses gan.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from magenta.models.gansynth.lib import flags as lib_flags
from magenta.models.gansynth.lib import logger as lib_logger
from magenta.models.gansynth.lib.eval import gan

slim = tf.contrib.slim

def set_flags(flags):
  """Set default flags."""
  flags.set_if_empty('batch_size', 64)
  flags.set_if_empty('gan', lib_flags.Flags())
  flags.set_if_empty('iters', 10*1000)
  flags.set_if_empty('log', False)
  flags.set_if_empty('output_dir', None)
  flags.set_if_empty('big_arch', False)
  flags.gan.set_if_empty('algorithm', 'wgan-gp')
  flags.gan.set_if_empty('lr_d', 2e-4)
  flags.gan.set_if_empty('lr_decay', 'linear')
  flags.gan.set_if_empty('l2_reg_d', 0.)
  flags.gan.set_if_empty('wgangp_minimax', True)
  gan.set_flags(flags.gan)


def _conv1d(x, dim, ksize, stride, **kwargs):
  kwargs['stride'] = (1, stride)
  x_shape = x.get_shape().as_list()
  x_shape_dynamic = tf.shape(x)
  x = tf.reshape(x, [x_shape_dynamic[0], 1, x_shape[1], x_shape[2]])
  x = slim.conv2d(x, dim, (1, ksize), **kwargs)
  x = tf.reshape(x, [x_shape_dynamic[0], x_shape[1] // stride, dim])
  return x


def run(flags, real_data, fake_data):
  """real_data and fake_data have shape [10000, 64000]."""

  with tf.Graph().as_default() as graph:
    step_placeholder = tf.placeholder(tf.int32, None)
    real_inputs = tf.placeholder(tf.float32, [None, 64000])
    fake_inputs = tf.placeholder(tf.float32, [None, 64000])
    reals = real_inputs[:, :, None]
    fakes = fake_inputs[:, :, None]

    def discriminator(x):
      """Discriminator network."""
      with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        padding = (65536-64000)//2
        out = tf.pad(x, [[0, 0], [padding, padding], [0, 0]])
        if flags.big_arch:
          layer_sizes = [64, 128, 128, 256, 256, 256]
        else:
          layer_sizes = [32, 64, 128, 256, 256]
        weights_initializer = tf.contrib.layers.variance_scaling_initializer()
        for i, layer_size in enumerate(layer_sizes):
          if i == 0:
            out = _conv1d(out, layer_size, 64, (4 if flags.big_arch else 16),
                          scope='conv{}'.format(i), normalizer_fn=None,
                          activation_fn=None,
                          weights_initializer=weights_initializer)
          else:
            out = _conv1d(out, layer_size, 8, 4,
                          scope='conv{}'.format(i), normalizer_fn=None,
                          activation_fn=None,
                          weights_initializer=weights_initializer)
          out = out*tf.nn.sigmoid(out)
        out = tf.reshape(out, [tf.shape(x)[0], 16*256])
        out = slim.fully_connected(out, 1, scope='fc',
                                   normalizer_fn=None, activation_fn=None)
        return out

    def generator(unused_z):
      return fakes

    discriminator(reals)  # initialize vars
    disc_params = slim.get_model_variables('discriminator')
    _, disc_cost, divergence = gan.losses(generator, discriminator, reals,
                                          None, disc_params, flags.gan)
    disc_train_op = gan.disc_train_op(disc_cost, disc_params, step_placeholder,
                                      flags.iters, flags.gan)

    def gen(samples):
      while True:
        np.random.shuffle(samples)
        for i in xrange(10*1000//flags.batch_size):
          yield samples[i*flags.batch_size:(i+1)*flags.batch_size]
    real_gen = gen(real_data)
    fake_gen = gen(fake_data)

    with tf.Session(graph=graph) as session:
      print('Training NN divergence model')
      if flags.log:
        logger = lib_logger.Logger(flags.output_dir)
      session.run(tf.variables_initializer(slim.get_variables()))
      # Train loop
      train_divs = []
      for step in xrange(flags.iters):
        start_time = time.time()
        real_inputs_ = real_gen.next()
        fake_inputs_ = fake_gen.next()
        disc_cost_, divergence_, _ = session.run(
            [disc_cost, divergence, disc_train_op],
            feed_dict={step_placeholder: step,
                       real_inputs: real_inputs_,
                       fake_inputs: fake_inputs_})
        train_divs.append(divergence_)
        if flags.log:
          logger.add_scalar('time', time.time() - start_time, step)
          logger.add_scalar('train_cost', disc_cost_, step)
          logger.add_scalar('train_divergence', divergence_, step)
          logger.print(step)
      if flags.log:
        logger.flush()

      return np.mean(train_divs[len(train_divs)//2:])
