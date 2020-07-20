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
"""Defines the SVGDecoder model."""

import copy

from magenta.contrib import rnn as contrib_rnn
from magenta.models.svg_vae import image_vae
from magenta.models.svg_vae import svg_decoder_loss
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import trainer_lib
import tensorflow.compat.v1 as tf

rnn = tf.nn.rnn_cell


@registry.register_model
class SVGDecoder(t2t_model.T2TModel):
  """Defines the SVGDecoder model."""

  def body(self, features):
    if self._hparams.initializer == 'orthogonal':
      raise ValueError('LSTM models fail with orthogonal initializer.')
    train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
    return self.render2cmd_v3_internal(features, self._hparams, train)

  def pretrained_visual_encoder(self, features, hparams):
    # we want the exact hparams used for training this vv
    vae_hparams = trainer_lib.create_hparams(
        hparams.vae_hparam_set, hparams.vae_hparams,
        data_dir=hparams.vae_data_dir, problem_name=hparams.vae_problem)

    # go back to root variable scope
    with tf.variable_scope(tf.VariableScope(tf.AUTO_REUSE, ''),
                           reuse=tf.AUTO_REUSE, auxiliary_name_scope=False):
      vae = image_vae.ImageVAE(vae_hparams, mode=self._hparams.mode,
                               problem_hparams=vae_hparams.problem_hparams)
      # the real input to vae will be features['rendered_targets']
      vae_features = copy.copy(features)
      vae_features['inputs'] = tf.reshape(vae_features['targets_psr'][:, -1, :],
                                          [-1, 64, 64, 1])
      vae_features['targets'] = vae_features['inputs']
      # we want vae to return bottleneck
      vae_features['bottleneck'] = tf.zeros((0, 128))
      sampled_bottleneck, _ = vae(vae_features)
      vae.initialize_from_ckpt(hparams.vae_ckpt_dir)

      if tf.executing_eagerly():
        sampled_bottleneck, _ = vae(vae_features)

    return sampled_bottleneck

  def render2cmd_v3_internal(self, features, hparams, train):
    # inputs and targets are both sequences with
    # shape = [batch, seq_len, 1, hparams.problem.feature_dim]
    targets = features['targets']
    losses = {}

    sampled_bottleneck = self.pretrained_visual_encoder(features, hparams)
    if hparams.sg_bottleneck:
      sampled_bottleneck = tf.stop_gradient(sampled_bottleneck)

    with tf.variable_scope('render2cmd_v3_internal'):
      # override bottleneck, or return it, if requested
      if 'bottleneck' in features:
        if common_layers.shape_list(features['bottleneck'])[0] == 0:
          # return sampled_bottleneck,
          # set losses['training'] = 0 so self.top() doesn't get called on it
          return sampled_bottleneck, {'training': 0.0}
        else:
          # we want to use the given bottleneck
          sampled_bottleneck = features['bottleneck']

      # finalize bottleneck
      unbottleneck_dim = hparams.hidden_size * 2  # twice because using LSTM
      if hparams.twice_decoder:
        unbottleneck_dim = unbottleneck_dim * 2

      # unbottleneck back to LSTMStateTuple
      dec_initial_state = []
      for hi in range(hparams.num_hidden_layers):
        unbottleneck = self.unbottleneck(sampled_bottleneck, unbottleneck_dim,
                                         name_append='_{}'.format(hi))
        dec_initial_state.append(
            rnn.LSTMStateTuple(
                c=unbottleneck[:, :unbottleneck_dim // 2],
                h=unbottleneck[:, unbottleneck_dim // 2:]))

      dec_initial_state = tuple(dec_initial_state)

      shifted_targets = common_layers.shift_right(targets)
      # Add 1 to account for the padding added to the left from shift_right
      targets_length = common_layers.length_from_embedding(shifted_targets) + 1

      # LSTM decoder
      hparams_decoder = copy.copy(hparams)
      if hparams.twice_decoder:
        hparams_decoder.hidden_size = 2 * hparams.hidden_size

      if hparams.mode == tf.estimator.ModeKeys.PREDICT:
        decoder_outputs, _ = self.lstm_decoder_infer(
            common_layers.flatten4d3d(shifted_targets),
            targets_length, hparams_decoder, features['targets_cls'],
            train, initial_state=dec_initial_state,
            bottleneck=sampled_bottleneck)
      else:
        decoder_outputs, _ = self.lstm_decoder(
            common_layers.flatten4d3d(shifted_targets),
            targets_length, hparams_decoder, features['targets_cls'],
            train, initial_state=dec_initial_state,
            bottleneck=sampled_bottleneck)

      ret = tf.expand_dims(decoder_outputs, axis=2)

    return ret, losses

  def lstm_decoder_infer(self, inputs, sequence_length, hparams, clss, train,
                         initial_state=None, bottleneck=None):
    # IN PREDICT MODE, RUN tf.while RNN
    max_decode_length = 51
    batch_size = common_layers.shape_list(inputs)[0]
    zero_pad, logits_so_far = self.create_initial_input_for_decode(batch_size)

    layers = rnn.MultiRNNCell([
        self.lstm_cell(hparams, train) for _ in range(hparams.num_hidden_layers)
    ])

    if initial_state is None:
      raise Exception('initial state should be init from bottleneck!')

    # append one-hot class to bottleneck, which will be given per step
    clss = tf.reshape(clss, [-1])
    if not hparams.use_cls:
      clss = tf.zeros_like(clss)
    if hparams.condition_on_sln:
      sln = tf.reshape(sequence_length, [-1])
      bottleneck = tf.concat((bottleneck,
                              tf.one_hot(clss, hparams.num_categories),
                              tf.one_hot(sln, max_decode_length)), -1)
    else:
      bottleneck = tf.concat((bottleneck,
                              tf.one_hot(clss, hparams.num_categories)), -1)

    def infer_step(logits_so_far, current_hidden):
      """Inference step of LSTM while loop."""
      # unflatten hidden:
      current_hidden = tuple(rnn.LSTMStateTuple(c=s[0], h=s[1])
                             for s in current_hidden)

      # put logits_so_far through top
      tm = self._problem_hparams.modality['targets']
      # need to reuse top params
      reset_scope = tf.variable_scope(tf.VariableScope(tf.AUTO_REUSE, ''),
                                      reuse=tf.AUTO_REUSE,
                                      auxiliary_name_scope=False)
      top_scope = tf.variable_scope('svg_decoder/{}_modality'.format(tm),
                                    reuse=tf.AUTO_REUSE)
      with reset_scope, top_scope:
        samples_so_far = self.hparams.top['targets'](
            logits_so_far, None, self.hparams, self.problem_hparams.vocab_size)
      # append a zero pad to the samples. this effectively shifts the samples
      # right, but, unlike shift_right, by not removing the last element, we
      # allow an empty samples_so_far to not be empty after padding
      samples_so_far = tf.concat([zero_pad, samples_so_far], axis=1)
      shifted_targets = common_layers.flatten4d3d(samples_so_far)
      # now take the very last one here, will be the actual input to the rnn
      shifted_targets = shifted_targets[:, -1:, :]

      # tile and append the bottleneck to inputs
      sln_offset = 0
      if hparams.condition_on_sln:
        sln_offset = 51
      pre_tile_y = tf.reshape(
          bottleneck,
          [common_layers.shape_list(bottleneck)[0], 1,
           hparams.bottleneck_bits + hparams.num_categories + sln_offset])
      overlay_x = tf.tile(pre_tile_y,
                          [1, common_layers.shape_list(shifted_targets)[1], 1])
      inputs = tf.concat([shifted_targets, overlay_x], -1)

      seq_len_batch = tf.ones([common_layers.shape_list(inputs)[0]])

      # RUN PRE-LSTM LAYER
      with tf.variable_scope('pre_decoder', reuse=tf.AUTO_REUSE):
        inputs = tf.layers.dense(inputs, hparams.hidden_size, name='bottom')
        inputs = tf.nn.tanh(inputs)

      # RUN LSTM
      with tf.variable_scope('lstm_decoder', reuse=tf.AUTO_REUSE):
        next_step, next_state = tf.nn.dynamic_rnn(
            layers, inputs, seq_len_batch, initial_state=current_hidden,
            dtype=tf.float32, time_major=False)

      next_step = tf.expand_dims(next_step, [1])
      logits_so_far = tf.concat([logits_so_far, next_step], 1)

      # flatten state
      next_state = tuple((s.c, s.h) for s in next_state)

      return logits_so_far, next_state

    def while_exit_cond(logits_so_far, unused_current_hidden):
      length = common_layers.shape_list(logits_so_far)[1]
      return length < max_decode_length

    # passing state must be flattened:
    initial_state = tuple([(s.c, s.h) for s in initial_state])

    # actually run tf.while:
    logits, final_state = tf.while_loop(
        while_exit_cond, infer_step,
        [logits_so_far, initial_state],
        shape_invariants=[
            tf.TensorShape([None, None, 1, hparams.hidden_size]),
            tuple([(s[0].get_shape(), s[1].get_shape())
                   for s in initial_state]),
        ],
        back_prop=False,
        parallel_iterations=1
    )

    # logits should be returned in 3d mode:
    logits = common_layers.flatten4d3d(logits)

    return logits, final_state

  def lstm_decoder(self, inputs, sequence_length, hparams, clss, train,
                   initial_state=None, bottleneck=None):
    # NOT IN PREDICT MODE. JUST RUN TEACHER-FORCED RNN:
    layers = rnn.MultiRNNCell([
        self.lstm_cell(hparams, train) for _ in range(hparams.num_hidden_layers)
    ])

    # append one-hot class to bottleneck, which will be given per step
    clss = tf.reshape(clss, [-1])
    if not hparams.use_cls:
      clss = tf.zeros_like(clss)
    if hparams.condition_on_sln:
      sln = tf.reshape(sequence_length, [-1])
      bottleneck = tf.concat((bottleneck,
                              tf.one_hot(clss, hparams.num_categories),
                              tf.one_hot(sln, 51)), -1)
    else:
      bottleneck = tf.concat((bottleneck,
                              tf.one_hot(clss, hparams.num_categories)), -1)

    # tile and append the bottleneck to inputs
    sln_offset = 0
    if hparams.condition_on_sln:
      sln_offset = 51
    pre_tile_y = tf.reshape(
        bottleneck,
        [common_layers.shape_list(bottleneck)[0], 1,
         hparams.bottleneck_bits + hparams.num_categories + sln_offset])
    overlay_x = tf.tile(pre_tile_y, [1, common_layers.shape_list(inputs)[1], 1])
    inputs = tf.concat([inputs, overlay_x], -1)

    with tf.variable_scope('pre_decoder', reuse=tf.AUTO_REUSE):
      inputs = tf.layers.dense(inputs, hparams.hidden_size, name='bottom')
      inputs = tf.nn.tanh(inputs)

    with tf.variable_scope('lstm_decoder', reuse=tf.AUTO_REUSE):
      return tf.nn.dynamic_rnn(
          layers, inputs, sequence_length, initial_state=initial_state,
          dtype=tf.float32, time_major=False)

  def lstm_cell(self, hparams, train):
    keep_prob = 1.0 - hparams.rec_dropout * tf.to_float(train)

    recurrent_dropout_cell = contrib_rnn.LayerNormBasicLSTMCell(
        hparams.hidden_size,
        layer_norm=hparams.layer_norm,
        dropout_keep_prob=keep_prob)

    if hparams.ff_dropout:
      return rnn.DropoutWrapper(
          recurrent_dropout_cell, input_keep_prob=keep_prob)
    return recurrent_dropout_cell

  def unbottleneck(self, x, res_size, reuse=tf.AUTO_REUSE, name_append=''):
    with tf.variable_scope('unbottleneck{}'.format(name_append), reuse=reuse):
      x = tf.layers.dense(x, res_size, name='dense', activation='tanh')
      return x

  def create_initial_input_for_decode(self, batch_size):
    # Create an initial output tensor. This will be passed
    # to the infer_step, which adds one timestep at every iteration.
    dim = self._problem_hparams.vocab_size['targets']
    hdim = self._hparams.hidden_size
    initial_output = tf.zeros((batch_size, 0, 1, hdim), dtype=tf.float32)
    zero_pad = tf.zeros((batch_size, 1, 1, dim), dtype=tf.float32)
    # Hack: foldl complains when the output shape is less specified than the
    # input shape, so we confuse it about the input shape.
    initial_output = tf.slice(initial_output, [0, 0, 0, 0],
                              common_layers.shape_list(initial_output))
    zero_pad = tf.slice(zero_pad, [0, 0, 0, 0],
                        common_layers.shape_list(zero_pad))
    return zero_pad, initial_output

  def _greedy_infer(self, features, extra_decode_length, use_tpu=False):
    # extra_decode_length is set to 0, unused.
    del extra_decode_length
    infer_features = copy.copy(features)
    if 'targets' not in infer_features:
      infer_features['targets'] = infer_features['infer_targets']
    logits, losses = self(infer_features)  # pylint: disable=not-callable
    return {
        'outputs': logits,
        'scores': None,
        'logits': logits,
        'losses': losses,
    }


@registry.register_hparams
def svg_decoder():
  """Basic hparams for SVG decoder."""
  hparams = common_hparams.basic_params1()
  hparams.daisy_chain_variables = False
  hparams.batch_size = 128
  hparams.hidden_size = 1024
  hparams.num_hidden_layers = 2
  hparams.initializer = 'uniform_unit_scaling'
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  hparams.num_hidden_layers = 4
  hparams.force_full_predict = True
  hparams.dropout = 0.5
  hparams.learning_rate_warmup_steps = 100000

  # LSTM-specific hparams
  hparams.add_hparam('vocab_size', None)

  # VAE params
  hparams.add_hparam('bottleneck_bits', 32)

  # VAE loss params (don't matter but must be defined)
  hparams.add_hparam('kl_beta', 300)
  hparams.add_hparam('free_bits_div', 4)

  # loss params
  hparams.add_hparam('soft_k', 10)
  hparams.add_hparam('mdn_k', 1)

  # params required by LayerNormLSTMCell, for us to just use recurrent dropout
  hparams.add_hparam('layer_norm', False)
  hparams.add_hparam('ff_dropout', True)
  hparams.add_hparam('rec_dropout', 0.3)

  # Decode architecture hparams
  hparams.add_hparam('twice_decoder', False)
  hparams.add_hparam('sg_bottleneck', True)
  hparams.add_hparam('condition_on_sln', False)
  hparams.add_hparam('use_cls', True)

  # MDN loss hparams
  hparams.add_hparam('num_mixture', 50)
  hparams.add_hparam('mix_temperature', 0.0001)
  hparams.add_hparam('gauss_temperature', 0.0001)
  hparams.add_hparam('dont_reduce_loss', False)

  # VAE meta hparams (to load image encoder)
  hparams.add_hparam('vae_ckpt_dir', '')
  hparams.add_hparam('vae_hparams', '')
  hparams.add_hparam('vae_data_dir', '')
  hparams.add_hparam('vae_hparam_set', 'image_vae')
  hparams.add_hparam('vae_problem', 'glyph_azzn_problem')

  # data format hparams
  hparams.add_hparam('num_categories', 62)

  # problem hparams (required, don't modify)
  hparams.add_hparam('absolute', False)
  hparams.add_hparam('just_render', False)
  hparams.add_hparam('plus_render', False)

  # modality hparams
  hparams.bottom = {
      'inputs': svg_decoder_loss.real_svg_bottom,
      'targets': svg_decoder_loss.real_svg_bottom,
  }
  hparams.top = {'targets': svg_decoder_loss.real_svg_top}
  hparams.loss = {'targets': svg_decoder_loss.real_svg_loss}

  return hparams
