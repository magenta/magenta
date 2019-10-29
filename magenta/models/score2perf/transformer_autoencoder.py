# Copyright 2019 The Magenta Authors.
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

"""Variations of Transformer autoencoder models for conditional music generation.

The Transformer autoencoder consists of an encoder and a decoder. The models
currently support conditioning on both performance and melody -- some things
needed to be hardcoded in order to get the model to train.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
from tensor2tensor.layers import transformer_layers
# pylint: disable=g-multiple-import
from tensor2tensor.models.transformer import (
    Transformer,
    transformer_decoder,
    transformer_prepare_encoder,
    transformer_prepare_decoder,
    features_to_nonpadding,
    _init_transformer_cache,
)
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import registry

import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import

# Alias some commonly reused layers, here and elsewhere.
transformer_prepare_encoder = transformer_layers.transformer_prepare_encoder
transformer_encoder = transformer_layers.transformer_encoder
transformer_ffn_layer = transformer_layers.transformer_ffn_layer


def baseline_perf_transformer_encode(encoder_function, inputs, target_space,
                                     hparams, attention_weights=None,
                                     features=None, losses=None,
                                     prepare_encoder_fn=None, **kwargs):
  """Encoding for baseline performance transformer, no mean-aggregation.

  Args:
    encoder_function: the encoder function
    inputs: Transformer inputs [batch_size, input_length, 1, hidden_dim] which
      will be flattened along the two spatial dimensions.
    target_space: scalar, target space ID.
    hparams: hyperparameters for model.
    attention_weights: weight to store attention to.
    features: optionally pass the entire features dictionary as well. This is
      needed now for "packed" datasets.
    losses: optional list onto which to append extra training losses
    prepare_encoder_fn: optional, alternative to transformer_prepare_encoder.
    **kwargs: additional arguments to pass to encoder_function

  Returns:
    Tuple of:
        encoder_output: Encoder representation.
            [batch_size, input_length, hidden_dim]
        encoder_decoder_attention_bias: Bias and mask weights for
            encoder-decoder attention. [batch_size, input_length]
  """
  inputs = common_layers.flatten4d3d(inputs)

  if not prepare_encoder_fn:
    prepare_encoder_fn = transformer_prepare_encoder
  encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
      prepare_encoder_fn(
          inputs, target_space, hparams, features=features,
          reuse_target_embedding=tf.AUTO_REUSE))

  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
      value=hparams.layer_prepostprocess_dropout,
      hparams=hparams)

  encoder_input = tf.nn.dropout(encoder_input,
                                1.0 - hparams.layer_prepostprocess_dropout)

  attn_bias_for_padding = None
  # Otherwise the encoder will just use encoder_self_attention_bias.
  if hparams.unidirectional_encoder:
    attn_bias_for_padding = encoder_decoder_attention_bias

  encoder_output = encoder_function(
      encoder_input,
      self_attention_bias,
      hparams,
      name="encoder",
      nonpadding=features_to_nonpadding(features, "inputs"),
      save_weights_to=attention_weights,
      make_image_summary=not common_layers.is_xla_compiled(),
      losses=losses,
      attn_bias_for_padding=attn_bias_for_padding,
      **kwargs)

  # no aggregation --> just return everything normally
  return encoder_output, encoder_decoder_attention_bias


def perf_transformer_encode(encoder_function, inputs, target_space, hparams,
                            attention_weights=None, features=None, losses=None,
                            prepare_encoder_fn=None, **kwargs):
  """Encoding for performance autoencoder, which mean-aggregates across time.

  Args:
    encoder_function: the encoder function
    inputs: Transformer inputs [batch_size, input_length, 1, hidden_dim] which
      will be flattened along the two spatial dimensions.
    target_space: scalar, target space ID.
    hparams: hyperparameters for model.
    attention_weights: weight to store attention to.
    features: optionally pass the entire features dictionary as well. This is
      needed now for "packed" datasets.
    losses: optional list onto which to append extra training losses
    prepare_encoder_fn: optional, alternative to transformer_prepare_encoder.
    **kwargs: additional arguments to pass to encoder_function

  Returns:
    Tuple of:
        encoder_output: Encoder representation.
            [batch_size, input_length, hidden_dim]
        encoder_decoder_attention_bias: Bias and mask weights for
            encoder-decoder attention. [batch_size, input_length]
  """
  inputs = common_layers.flatten4d3d(inputs)

  if not prepare_encoder_fn:
    prepare_encoder_fn = transformer_prepare_encoder
  encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
      prepare_encoder_fn(
          inputs, target_space, hparams, features=features,
          reuse_target_embedding=tf.AUTO_REUSE))

  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
      value=hparams.layer_prepostprocess_dropout,
      hparams=hparams)

  encoder_input = tf.nn.dropout(encoder_input,
                                1.0 - hparams.layer_prepostprocess_dropout)

  attn_bias_for_padding = None
  # Otherwise the encoder will just use encoder_self_attention_bias.
  if hparams.unidirectional_encoder:
    attn_bias_for_padding = encoder_decoder_attention_bias

  encoder_output = encoder_function(
      encoder_input,
      self_attention_bias,
      hparams,
      name="encoder",
      nonpadding=features_to_nonpadding(features, "inputs"),
      save_weights_to=attention_weights,
      make_image_summary=not common_layers.is_xla_compiled(),
      losses=losses,
      attn_bias_for_padding=attn_bias_for_padding,
      **kwargs)

  encoder_output = tf.math.reduce_mean(
      encoder_output, axis=1, keep_dims=True)
  encoder_decoder_attention_bias = tf.math.reduce_mean(
      encoder_decoder_attention_bias, axis=-1, keep_dims=True)

  return encoder_output, encoder_decoder_attention_bias


def transformer_decode(decoder_function,
                       decoder_input,
                       encoder_output,
                       encoder_decoder_attention_bias,
                       decoder_self_attention_bias,
                       hparams,
                       attention_weights=None,
                       cache=None,
                       decode_loop_step=None,
                       nonpadding=None,
                       losses=None,
                       **kwargs):
  """Decode Transformer outputs from encoder representation.

  Args:
    decoder_function: the decoder function
    decoder_input: inputs to bottom of the model. [batch_size, decoder_length,
      hidden_dim]
    encoder_output: Encoder representation. [batch_size, input_length,
      hidden_dim]
    encoder_decoder_attention_bias: Bias and mask weights for encoder-decoder
      attention. [batch_size, input_length]
    decoder_self_attention_bias: Bias and mask weights for decoder
      self-attention. [batch_size, decoder_length]
    hparams: hyperparameters for model.
    attention_weights: weight to store attention to.
    cache: dict, containing tensors which are the results of previous
      attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop. Only used
      for inference on TPU.
    nonpadding: optional Tensor with shape [batch_size, decoder_length]
    losses: optional list onto which to append extra training losses
    **kwargs: additional arguments to pass to decoder_function

  Returns:
    Final decoder representation. [batch_size, decoder_length, hidden_dim]
  """
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
      value=hparams.layer_prepostprocess_dropout,
      hparams=hparams)
  decoder_input = tf.nn.dropout(decoder_input,
                                1.0 - hparams.layer_prepostprocess_dropout)

  decoder_output = decoder_function(
      decoder_input,
      encoder_output,
      decoder_self_attention_bias,
      encoder_decoder_attention_bias,
      hparams,
      cache=cache,
      decode_loop_step=decode_loop_step,
      nonpadding=nonpadding,
      save_weights_to=attention_weights,
      losses=losses,
      **kwargs)

  if (common_layers.is_xla_compiled() and
      hparams.mode == tf.estimator.ModeKeys.TRAIN):
    # TPU does not react kindly to extra dimensions.
    # TODO(noam): remove this once TPU is more forgiving of extra dims.
    return decoder_output
  else:
    # Expand since t2t expects 4d tensors.
    return tf.expand_dims(decoder_output, axis=2)


@registry.register_model
class PerformanceTransformer(Transformer):
  """Transformer Autoencoder, which uses a single performance encoding."""

  def __init__(self, *args, **kwargs):
    super(PerformanceTransformer, self).__init__(*args, **kwargs)
    self.attention_weights = {}  # For visualizing attention heads.
    self.recurrent_memory_by_layer = None  # Override to enable recurrent memory
    self._encoder_function = transformer_encoder
    self._decoder_function = transformer_decoder
    self._init_cache_fn = _init_transformer_cache
    self._prepare_encoder_fn = transformer_prepare_encoder
    self._prepare_decoder_fn = transformer_prepare_decoder

  def encode(self, inputs, target_space, hparams,
             features=None, losses=None):
    """Encode transformer inputs, see transformer_encode."""
    return perf_transformer_encode(
        self._encoder_function, inputs, target_space, hparams,
        attention_weights=self.attention_weights,
        features=features, losses=losses,
        prepare_encoder_fn=self._prepare_encoder_fn)


@registry.register_model
class BaselinePerformanceTransformer(PerformanceTransformer):
  """Performance Transformer Autoencoder, without mean-aggregation."""

  def __init__(self, *args, **kwargs):
    super(BaselinePerformanceTransformer, self).__init__(*args, **kwargs)
    self.attention_weights = {}  # For visualizing attention heads.
    self.recurrent_memory_by_layer = None  # Override to enable recurrent memory
    self._encoder_function = transformer_encoder
    self._decoder_function = transformer_decoder
    self._init_cache_fn = _init_transformer_cache
    self._prepare_encoder_fn = transformer_prepare_encoder
    self._prepare_decoder_fn = transformer_prepare_decoder

  @property
  def has_input(self):
    if self._problem_hparams:
      modalities = self._problem_hparams.modality
      return ("performance" in modalities) or ("inputs" in modalities)
    else:
      return True

  def encode(self, inputs, target_space, hparams,
             features=None, losses=None):
    """Encode transformer inputs, see transformer_encode."""
    return baseline_perf_transformer_encode(
        self._encoder_function, inputs, target_space, hparams,
        attention_weights=self.attention_weights,
        features=features, losses=losses,
        prepare_encoder_fn=self._prepare_encoder_fn)
