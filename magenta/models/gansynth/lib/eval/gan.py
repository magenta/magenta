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
"""Very minimal GAN library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def set_flags(flags):
  """Set default flags."""
  flags.set_if_empty('algorithm', 'vanilla')
  flags.set_if_empty('architecture', 'dcgan')
  flags.set_if_empty('dim', 64)
  flags.set_if_empty('dim_z', 128)
  flags.set_if_empty('extra_depth', 0)
  flags.set_if_empty('initializer_d', 'xavier')
  flags.set_if_empty('lr_decay', 'none')
  flags.set_if_empty('nonlinearity', 'default')
  flags.set_if_empty('norm', True)
  flags.set_if_empty('l2_reg_d', 1e-3)
  flags.set_if_empty('weight_clip_d', -1)
  flags.set_if_empty('weight_decay_g', None)
  flags.set_if_empty('weight_decay_d', None)
  flags.set_if_empty('z_seed', None)
  if flags.algorithm in ['wgan-gp', 'wgan-lp', 'wgan-v3', 'wgan-gp-quadratic']:
    flags.set_if_empty('lr', 1e-4)
    flags.set_if_empty('beta1', 0.)
    flags.set_if_empty('beta2', 0.9)
    flags.set_if_empty('disc_iters', 5)
    flags.set_if_empty('wgangp_lambda', 10)
    flags.set_if_empty('wgangp_minimax', False)
  elif flags.algorithm in ['vanilla', 'vanilla_minimax']:
    flags.set_if_empty('lr', 2e-4)
    flags.set_if_empty('beta1', 0.5)
    flags.set_if_empty('beta2', 0.999)
    flags.set_if_empty('disc_iters', 1)
  else:
    raise Exception('invalid gan flags.algorithm')
  flags.set_if_empty('dim_g', flags.dim)
  flags.set_if_empty('dim_d', flags.dim)
  flags.set_if_empty('extra_depth_g', flags.extra_depth)
  flags.set_if_empty('extra_depth_d', flags.extra_depth)
  flags.set_if_empty('lr_d', flags.lr)
  flags.set_if_empty('lr_g', flags.lr)
  flags.set_if_empty('nonlinearity_g', flags.nonlinearity)
  flags.set_if_empty('nonlinearity_d', flags.nonlinearity)
  flags.set_if_empty('norm_g', flags.norm)
  flags.set_if_empty('norm_d', flags.norm)


def random_latents(batch_size, flags, antithetic_sampling=False):
  if antithetic_sampling:
    half = tf.random_normal([batch_size//2, flags.dim_z], seed=flags.z_seed)
    return tf.concat([half, -half], axis=0)
  else:
    return tf.random_normal([batch_size, flags.dim_z], seed=flags.z_seed)


def _leaky_relu(x):
  return tf.maximum(0.2*x, x)


def _swish(x):
  return x*tf.nn.sigmoid(x)


def _softplus(x):
  return tf.nn.softplus(x)


def _elu_softplus(x):
  """softplus that looks roughly like elu but is smooth."""
  return (tf.nn.softplus((2*x)+2)/2)-1


def nonlinearity_fn(flag, is_discriminator):
  """Choose a nonlinearity based on flag string."""
  if flag == 'default':
    if is_discriminator:
      return _leaky_relu
    else:
      return tf.nn.relu
  elif flag == 'leaky_relu':
    return _leaky_relu
  elif flag == 'relu':
    return tf.nn.relu
  elif flag == 'elu':
    return tf.nn.elu
  elif flag == 'swish':
    return _swish
  elif flag == 'softplus':
    return _softplus
  elif flag == 'elu_softplus':
    return _elu_softplus
  elif flag == 'exp':
    return tf.exp
  elif flag == 'tanh':
    return tf.tanh
  elif flag == 'sigmoid':
    return tf.nn.sigmoid
  else:
    raise Exception('invalid nonlinearity {}'.format(flag))


def generator(z, flags, scope=None, reuse=None):
  if flags.architecture == 'dcgan':
    return dcgan_generator(z, flags, scope, reuse)
  # elif flags.architecture == 'resnet':
  #   return resnet_generator(z, flags, scope, reuse)


def discriminator(x, flags, scope=None, reuse=None):
  if flags.architecture == 'dcgan':
    return dcgan_discriminator(x, flags, scope, reuse)
  # elif flags.architecture == 'resnet':
  #   return resnet_discriminator(x, flags, scope, reuse)


def dcgan_generator(z, flags, scope=None, reuse=None):
  """Generator function."""
  nonlinearity = nonlinearity_fn(flags.nonlinearity_g, False)

  if not flags.norm_g:
    normalizer = None
  else:
    normalizer = slim.batch_norm

  with tf.variable_scope(scope, reuse=reuse):
    out = slim.fully_connected(z, 4*4*(4*flags.dim_g), scope='fc',
                               normalizer_fn=normalizer,
                               activation_fn=nonlinearity)
    out = tf.reshape(out, [-1, 4, 4, 4*flags.dim_g])
    out = slim.conv2d_transpose(out, 2*flags.dim_g, 5, scope='conv1', stride=2,
                                normalizer_fn=normalizer,
                                activation_fn=nonlinearity)

    for i in range(flags.extra_depth_g):
      out = slim.conv2d(out, 2*flags.dim_g, 3, scope='extraconv1.{}'.format(i),
                        normalizer_fn=normalizer, activation_fn=nonlinearity)

    out = slim.conv2d_transpose(out, flags.dim_g, 5, scope='conv2', stride=2,
                                normalizer_fn=normalizer,
                                activation_fn=nonlinearity)

    for i in range(flags.extra_depth_g):
      out = slim.conv2d(out, flags.dim_g, 3, scope='extraconv2.{}'.format(i),
                        normalizer_fn=normalizer, activation_fn=nonlinearity)

    out = slim.conv2d_transpose(out, 3, 5, scope='conv3', stride=2,
                                activation_fn=tf.tanh)
    return out


def dcgan_discriminator(x, flags, scope=None, reuse=None):
  """Discriminator function."""
  nonlinearity = nonlinearity_fn(flags.nonlinearity_d, True)

  with tf.variable_scope(scope, reuse=reuse):
    if not flags.norm_d:
      normalizer = None
    elif flags.algorithm == 'vanilla':
      normalizer = slim.batch_norm
    else:
      normalizer = slim.layer_norm

    if flags.initializer_d == 'xavier':
      initializer = tf.contrib.layers.xavier_initializer()
    elif flags.initializer_d == 'orth_gain2':
      initializer = tf.orthogonal_initializer(gain=2.)
    elif flags.initializer_d == 'he':
      initializer = tf.contrib.layers.variance_scaling_initializer()
    elif flags.initializer_d == 'he_uniform':
      initializer = tf.contrib.layers.variance_scaling_initializer(uniform=True)

    out = slim.conv2d(x, flags.dim_d, 5, scope='conv1', stride=2,
                      activation_fn=nonlinearity,
                      weights_initializer=initializer)

    for i in range(flags.extra_depth_d):
      out = slim.conv2d(out, flags.dim_d, 3, scope='extraconv1.{}'.format(i),
                        activation_fn=nonlinearity, normalizer_fn=normalizer,
                        weights_initializer=initializer)

    out = slim.conv2d(out, 2*flags.dim_d, 5, scope='conv2', stride=2,
                      activation_fn=nonlinearity,
                      normalizer_fn=normalizer, weights_initializer=initializer)

    for i in range(flags.extra_depth_d):
      out = slim.conv2d(out, 2*flags.dim_d, 3, scope='extraconv2.{}'.format(i),
                        activation_fn=nonlinearity,
                        normalizer_fn=normalizer,
                        weights_initializer=initializer)

    out = slim.conv2d(out, 4*flags.dim_d, 5, scope='conv3', stride=2,
                      activation_fn=nonlinearity,
                      normalizer_fn=normalizer, weights_initializer=initializer)
    out = tf.reshape(out, [-1, 4*4*(4*flags.dim_d)])
    out = slim.fully_connected(out, 1, scope='fc', activation_fn=None)
    return out


def losses(generator_fn, discriminator_fn, real_data, random_latents_,
           disc_params, flags):
  """Loss functions."""
  fake_data = generator_fn(random_latents_)
  # if flags.algorithm == 'wgan-gp':
  #   disc_all = discriminator_fn(tf.concat([real_data, fake_data], axis=0))
  #   len_real_data = tf.shape(real_data)[0]
  #   disc_real = disc_all[:len_real_data]
  #   disc_fake = disc_all[len_real_data:]
  # else:
  disc_real = discriminator_fn(real_data)
  disc_fake = discriminator_fn(fake_data)

  l2_reg_d_cost = 0.
  if flags.l2_reg_d > 0:
    for p in disc_params:
      if 'weights' in p.name:
        l2_reg_d_cost += tf.nn.l2_loss(p)
    l2_reg_d_cost *= flags.l2_reg_d

  if flags.algorithm == 'vanilla':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake, labels=tf.ones_like(disc_fake)))
    disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake, labels=tf.zeros_like(disc_fake)))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real, labels=tf.ones_like(disc_real)))
    # I believe this is KL(Pg||Pr)-2*JSD(Pg;Pr) (arjovsky & bottou 2017),
    # but it's probably missing a term because it's independent of Pr
    divergence = gen_cost
    disc_cost += l2_reg_d_cost

  elif flags.algorithm == 'vanilla_minimax':
    disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake, labels=tf.zeros_like(disc_fake)))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real, labels=tf.ones_like(disc_real)))
    gen_cost = -disc_cost
    divergence = ((-disc_cost)+tf.log(4.))/2.
    disc_cost += l2_reg_d_cost

  elif flags.algorithm == 'wgan-gp':
    input_ndim = len(real_data.get_shape())
    wgan_disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    alpha = tf.random_uniform(
        shape=[tf.shape(real_data)[0]] + [1 for i in xrange(input_ndim-1)],
        minval=0., maxval=1.)
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(discriminator_fn(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(
        tf.square(gradients),
        reduction_indices=[i for i in xrange(1, input_ndim)]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost = wgan_disc_cost + (flags.wgangp_lambda * gradient_penalty)
    disc_cost += l2_reg_d_cost
    if flags.wgangp_minimax:
      gen_cost = -disc_cost
      divergence = -disc_cost
    else:
      gen_cost = -tf.reduce_mean(disc_fake)
      divergence = -wgan_disc_cost

  elif flags.algorithm == 'wgan-lp':
    # max(0,||grad_d||-1)^2 penalty
    # Seems to work about as well as WGAN-GP; no strong evidence that it's
    # better or worse.

    wgan_disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    alpha = tf.random_uniform(shape=[tf.shape(real_data)[0], 1, 1, 1],
                              minval=0., maxval=1.)
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(discriminator_fn(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                   reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean(tf.maximum(0., slopes-1.)**2)
    disc_cost = wgan_disc_cost + (flags.wgangp_lambda * gradient_penalty)
    disc_cost += l2_reg_d_cost
    if flags.wgangp_minimax:
      gen_cost = -disc_cost
      divergence = -disc_cost
    else:
      gen_cost = -tf.reduce_mean(disc_fake)
      divergence = -wgan_disc_cost

  elif flags.algorithm == 'wgan-v3':
    # WGAN based on finite differences rather than gradients
    # This doesn't work nearly as well as WGAN-GP, but maybe one day it will...

    wgan_disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    differences = fake_data - real_data
    alpha = tf.random_uniform(shape=[tf.shape(real_data)[0]],
                              minval=0., maxval=1.)
    interpolates = real_data + (alpha[:, None, None, None] * differences)
    disc_interps = discriminator_fn(interpolates)
    distances = tf.sqrt(tf.reduce_sum(differences**2, axis=[1, 2, 3]))

    penalty1 = tf.maximum(
        1.,
        tf.abs(disc_real - disc_interps) / (alpha*distances)[:, None]) - 1.
    penalty2 = tf.maximum(
        1.,
        tf.abs(disc_fake - disc_interps) / ((1-alpha)*distances)[:, None]) - 1.

    disc_cost = wgan_disc_cost
    disc_cost += (tf.reduce_mean(penalty1**2)) + (tf.reduce_mean(penalty2**2))
    disc_cost += l2_reg_d_cost
    gen_cost = -disc_cost
    divergence = -disc_cost

  elif flags.algorithm == 'wgan-gp-quadratic':
    # max(0,||grad_d||-distance)^2 penalty
    wgan_disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    differences = fake_data - real_data
    alpha = tf.random_uniform(shape=[tf.shape(real_data)[0]],
                              minval=0., maxval=1.)
    interpolates = real_data + (alpha[:, None, None, None]*differences)
    distances = tf.sqrt(tf.reduce_sum(differences**2, axis=[1, 2, 3]))
    interp_dists = alpha*distances
    gradients = tf.gradients(discriminator_fn(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                   reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean(tf.maximum(0., (slopes-interp_dists)**2))
    disc_cost = wgan_disc_cost + (flags.wgangp_lambda * gradient_penalty)
    disc_cost += l2_reg_d_cost
    if flags.wgangp_minimax:
      gen_cost = -disc_cost
      divergence = -disc_cost
    else:
      gen_cost = -tf.reduce_mean(disc_fake)
      divergence = -wgan_disc_cost

  return gen_cost, disc_cost, divergence


def gen_train_op(cost, params, step, iters, flags):
  """Generator training operation."""
  if flags.lr_decay == 'linear':
    step_lr = (1.-(tf.cast(step, tf.float32)/iters))
  elif flags.lr_decay == 'quadratic':
    step_lr = ((1.-(tf.cast(step, tf.float32)/iters))**2)
  elif flags.lr_decay == 'none':
    step_lr = 1.
  train_op = tf.train.AdamOptimizer(
      step_lr*flags.lr_g, flags.beta1, flags.beta2).minimize(
          cost, var_list=params, colocate_gradients_with_ops=True)

  if flags.weight_decay_g is not None:
    decay = (step_lr*flags.weight_decay_g)
    with tf.control_dependencies([train_op]):
      weights = [p for p in params if 'weights' in p.name]
      decayed = [w-(decay*w) for w in weights]
      decay_op = tf.group(*[tf.assign(w, d) for w, d in zip(weights, decayed)])
    train_op = decay_op

  return train_op


def disc_train_op(cost, params, step, iters, flags):
  """Discriminator training operation."""
  if flags.lr_decay == 'linear':
    step_lr = (1.-(tf.cast(step, tf.float32)/iters))
  elif flags.lr_decay == 'quadratic':
    step_lr = ((1.-(tf.cast(step, tf.float32)/iters))**2)
  elif flags.lr_decay == 'drop_after_90k':
    step_lr = tf.cond(step > 90000, lambda: 0.1, lambda: 1.0)
  elif flags.lr_decay == 'none':
    step_lr = 1.
  train_op = tf.train.AdamOptimizer(
      step_lr*flags.lr_d, flags.beta1, flags.beta2).minimize(
          cost, var_list=params, colocate_gradients_with_ops=True)

  if flags.weight_decay_d is not None:
    decay = (step_lr*flags.weight_decay_d)
    with tf.control_dependencies([train_op]):
      weights = [p for p in params if 'weights' in p.name]
      decayed = [w-(decay*w) for w in weights]
      decay_op = tf.group(*[tf.assign(w, d) for w, d in zip(weights, decayed)])
    train_op = decay_op

  if flags.weight_clip_d >= 0:
    # Clip *all* the params, like the original WGAN implementation
    clip = flags.weight_clip_d
    with tf.control_dependencies([train_op]):
      clipped = [tf.clip_by_value(p, -clip, clip) for p in params]
      clip_op = tf.group(*[tf.assign(p, c) for c, p in zip(clipped, params)])
    train_op = clip_op

  return train_op
