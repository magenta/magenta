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

"""Defines SVG decoder loss."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
import tensorflow.compat.v1 as tf


# pylint: disable=redefined-outer-name
# pylint: disable=unused-variable
_summarized_losses = False  # pylint: disable=invalid-name


def _tf_lognormal(y, mean, logstd, logsqrttwopi):
  return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - logsqrttwopi


def _get_mdn_loss(logmix, mean, logstd, y, batch_mask, dont_reduce_loss):
  """Computes MDN loss term for svg decoder model."""
  logsqrttwopi = np.log(np.sqrt(2.0 * np.pi))

  v = logmix + _tf_lognormal(y, mean, logstd, logsqrttwopi)
  v = tf.reduce_logsumexp(v, 1, keepdims=True)
  v = tf.reshape(v, [-1, 51, 1, 6])

  # mask out unimportant terms given the ground truth commands
  v = tf.multiply(v, batch_mask)
  if dont_reduce_loss:
    return -tf.reduce_mean(tf.reduce_sum(v, axis=3), [1, 2])
  return -tf.reduce_mean(tf.reduce_sum(v, axis=3))


def real_svg_loss(top_out, targets, model_hparams, unused_vocab_size,
                  unused_weights_fn):
  """Computes loss for svg decoder model."""
  # targets already come in 10-dim mode, no need to so any mdn stuff
  # obviously.
  targets_commands_rel = targets[..., :4]
  targets_args_rel = targets[..., 4:]

  with tf.variable_scope('full_command_loss'):
    num_mix = model_hparams.num_mixture
    commands = top_out[:, :, :, :4]
    args = top_out[:, :, :, 4:]
    # args are [batch, seq, 1, 6*3*num_mix]. want [batch * seq * 6, 3*num_mix]
    args = tf.reshape(args, [-1, 3 * num_mix])
    out_logmix, out_mean, out_logstd = _get_mdn_coef(args)

    # before we compute mdn_args_loss, we need to create a mask for elements
    # to ignore on it.
    # create mask
    masktemplate = tf.constant([[0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 1.],
                                [0., 0., 0., 0., 1., 1.],
                                [1., 1., 1., 1., 1., 1.]])
    mask = tf.tensordot(targets_commands_rel, masktemplate, [[-1], [-2]])

    # calculate mdn loss, which auto masks it out
    targs_flat = tf.reshape(targets_args_rel, [-1, 1])
    mdn_loss = _get_mdn_loss(out_logmix, out_mean, out_logstd, targs_flat, mask,
                             model_hparams.dont_reduce_loss)

    # we dont have to manually mask out the softmax xent loss because
    # internally, each dimention of the xent loss is multiplied by the
    # given probability in the label for that dim. So for a one-hot label [0,
    # 1, 0] the xent loss between logit[0] and label[0] are multiplied by 0,
    # whereas between logit[1] and label[1] are multiplied by 1. Because our
    # targets_commands_rel is all 0s for the padding, sofmax_xent_loss is 0
    # for those elements as well.
    softmax_xent_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=targets_commands_rel, logits=commands)

    # Accumulate losses
    if model_hparams.dont_reduce_loss:
      softmax_xent_loss = tf.reduce_mean(softmax_xent_loss, [1, 2])
    else:
      softmax_xent_loss = tf.reduce_mean(softmax_xent_loss)
    loss = (model_hparams.mdn_k  * mdn_loss +
            model_hparams.soft_k * softmax_xent_loss)

  global _summarized_losses
  if not _summarized_losses:
    with tf.name_scope(None), tf.name_scope('losses_command'):
      tf.summary.scalar('mdn_loss', mdn_loss)
      tf.summary.scalar('softmax_xent_loss', softmax_xent_loss)

  # this tells us not to re-create the summary ops
  _summarized_losses = True

  return loss, tf.constant(1.0)


def _get_mdn_coef(output):
  logmix, mean, logstd = tf.split(output, 3, -1)
  logmix = logmix - tf.reduce_logsumexp(logmix, -1, keepdims=True)
  return logmix, mean, logstd


@modalities.is_pointwise
def real_svg_top(body_output, unused_targets, model_hparams, unused_vocab_size,
                 hard=False):
  """Applies the Mixture Density Network on top of the LSTM outputs.

  Args:
    body_output: outputs from LSTM with shape [batch, seqlen, 1, hidden_size]
    unused_targets: what the ground truth SVG outputted should be (unused).
    model_hparams: hyper-parameters, should include num_mixture,
      mix_temperature, and gauss_temperature.
    unused_vocab_size: unused
    hard: whether to force predict mode functionality, or return all MDN
      components

  Returns:
    The MDN output. Could be shape [batch, seqlen, 1, 10] if in predict mode
      (or hard=True) or shape [batch, seqlen, 1, 4 + 6 * num_mix * 3], in train.
  """
  # mixture of gaussians for 6 args plus 4 extra states for cmds
  num_mix = model_hparams.num_mixture
  nout = 4 + 6 * num_mix * 3

  # the 'hard' option is meant to be used if 'top' is called within body
  with tf.variable_scope('real_top', reuse=tf.AUTO_REUSE):
    ret = tf.layers.dense(body_output, nout, name='top')
    batch_size = common_layers.shape_list(ret)[0]

    if hard or model_hparams.mode == tf.estimator.ModeKeys.PREDICT:
      temperature = model_hparams.mix_temperature

      # apply temperature, do softmax
      command = tf.identity(ret[:, :, :, :4]) / temperature
      command = tf.exp(command -
                       tf.reduce_max(command, axis=[-1], keepdims=True))
      command = command / tf.reduce_sum(command, axis=[-1], keepdims=True)

      # sample from the given probs, this is the same as get_pi_idx,
      # and already returns not soft prob
      command = tf.distributions.Categorical(probs=command).sample()
      # this is now [batch, seq, 1], need to make it one_hot
      command = tf.one_hot(command, 4)

      arguments = ret[:, :, :, 4:]
      # args are [batch, seq, 1, 6*3*num_mix]. want [batch * seq * 6, 3*num_mix]
      arguments = tf.reshape(arguments, [-1, 3 * num_mix])

      out_logmix, out_mean, out_logstd = _get_mdn_coef(arguments)
      # these are [batch*seq*6, num_mix]

      # apply temp to logmix
      out_logmix = tf.identity(out_logmix) / temperature
      out_logmix = tf.exp(out_logmix -
                          tf.reduce_max(out_logmix, axis=[-1], keepdims=True))
      out_logmix = out_logmix / tf.reduce_sum(
          out_logmix, axis=[-1], keepdims=True)
      # get_pi_idx
      out_logmix = tf.distributions.Categorical(probs=out_logmix).sample()
      # should now be [batch*seq*6, 1]
      out_logmix = tf.cast(out_logmix, tf.int32)
      out_logmix = tf.reshape(out_logmix, [-1])
      # prepare for gather
      out_logmix = tf.stack(
          [tf.range(tf.size(out_logmix)), out_logmix], axis=-1)

      chosen_mean = tf.gather_nd(out_mean, out_logmix)
      chosen_logstd = tf.gather_nd(out_logstd, out_logmix)

      # sample!!
      rand_gaussian = (tf.random.normal(tf.shape(chosen_mean)) *
                       tf.sqrt(model_hparams.gauss_temperature))
      arguments = chosen_mean + tf.exp(chosen_logstd) * rand_gaussian
      arguments = tf.reshape(arguments, [batch_size, -1, 1, 6])

      # concat with the command we picked!
      ret = tf.concat([command, arguments], axis=-1)

  return ret


def real_svg_bottom(features, unused_model_hparams, unused_vocab_size):
  with tf.variable_scope('real_bottom', reuse=tf.AUTO_REUSE):
    return tf.identity(features)
