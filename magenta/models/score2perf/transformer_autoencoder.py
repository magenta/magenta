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
"""Variations of Transformer autoencoder for conditional music generation.

The Transformer autoencoder consists of an encoder and a decoder. The models
currently support conditioning on both performance and melody -- some things
needed to be hardcoded in order to get the model to train.
"""

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
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

import tensorflow.compat.v1 as tf

# pylint: disable=g-direct-tensorflow-import

# Alias some commonly reused layers, here and elsewhere.
transformer_prepare_encoder = transformer_layers.transformer_prepare_encoder
transformer_encoder = transformer_layers.transformer_encoder
transformer_ffn_layer = transformer_layers.transformer_ffn_layer


def perf_transformer_encode(encoder_function, inputs, target_space, hparams,
                            baseline, attention_weights=None, features=None,
                            losses=None, prepare_encoder_fn=None, **kwargs):
  """Encoding for performance autoencoder, which mean-aggregates across time.

  Args:
    encoder_function: the encoder function
    inputs: Transformer inputs [batch_size, input_length, 1, hidden_dim] which
      will be flattened along the two spatial dimensions.
    target_space: scalar, target space ID.
    hparams: hyperparameters for model.
    baseline: if True, does not mean-aggregate the encoder output.
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

  if not baseline:
    encoder_output = tf.math.reduce_mean(
        encoder_output, axis=1, keep_dims=True)
    encoder_decoder_attention_bias = tf.math.reduce_mean(
        encoder_decoder_attention_bias, axis=-1, keep_dims=True)

  return encoder_output, encoder_decoder_attention_bias


def mel_perf_transformer_encode(encoder_function, perf_inputs, mel_inputs,
                                target_space, hparams, attention_weights=None,
                                features=None, losses=None,
                                prepare_encoder_fn=None, **kwargs):
  """Encode transformer inputs. Used for melody & performance autoencoder.

  Performance is mean-aggregated across time and combined with melody in a
  variety of different ways.

  Args:
    encoder_function: the encoder function
    perf_inputs: Transformer inputs [batch_size, input_length, 1, hidden_dim]
    which will be flattened along the two spatial dimensions.
    mel_inputs: Transformer inputs [batch_size, input_length, 1, hidden_dim]
    which will be flattened along the two spatial dimensions.
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
  perf_inputs = common_layers.flatten4d3d(perf_inputs)
  mel_inputs = common_layers.flatten4d3d(mel_inputs)

  if not prepare_encoder_fn:
    prepare_encoder_fn = transformer_prepare_encoder
  perf_encoder_input, perf_self_attention_bias, perf_encdec_attention_bias = (
      prepare_encoder_fn(
          perf_inputs, target_space, hparams, features=features,
          reuse_target_embedding=tf.AUTO_REUSE))

  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
      value=hparams.layer_prepostprocess_dropout,
      hparams=hparams)

  perf_encoder_input = tf.nn.dropout(perf_encoder_input,
                                     1.0 - hparams.layer_prepostprocess_dropout)

  perf_attn_bias_for_padding = None
  # Otherwise the encoder will just use encoder_self_attention_bias.
  if hparams.unidirectional_encoder:
    perf_attn_bias_for_padding = perf_encdec_attention_bias

  # do the same thing for melody
  mel_encoder_input, mel_self_attention_bias, mel_encdec_attention_bias = (
      prepare_encoder_fn(
          mel_inputs, target_space, hparams, features=features,
          reuse_target_embedding=tf.AUTO_REUSE))

  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
      value=hparams.layer_prepostprocess_dropout,
      hparams=hparams)

  mel_encoder_input = tf.nn.dropout(mel_encoder_input,
                                    1.0 - hparams.layer_prepostprocess_dropout)

  mel_attn_bias_for_padding = None
  # Otherwise the encoder will just use encoder_self_attention_bias.
  if hparams.unidirectional_encoder:
    mel_attn_bias_for_padding = mel_encdec_attention_bias

  # use the proper encoder function for perf/melody
  perf_encoder_output = encoder_function(
      perf_encoder_input,
      perf_self_attention_bias,
      hparams,
      name="perf_encoder",
      nonpadding=features_to_nonpadding(features, "inputs"),
      save_weights_to=attention_weights,
      make_image_summary=not common_layers.is_xla_compiled(),
      losses=losses,
      attn_bias_for_padding=perf_attn_bias_for_padding,
      **kwargs)
  # same thing for melody
  mel_encoder_output = encoder_function(
      mel_encoder_input,
      mel_self_attention_bias,
      hparams,
      name="mel_encoder",
      nonpadding=features_to_nonpadding(features, "inputs"),
      save_weights_to=attention_weights,
      make_image_summary=not common_layers.is_xla_compiled(),
      losses=losses,
      attn_bias_for_padding=mel_attn_bias_for_padding,
      **kwargs)

  # concatenate the global mean vector/bias term with the full melody encoding
  perf_mean_vector = tf.math.reduce_mean(
      perf_encoder_output, axis=1, keep_dims=True)

  # different methods of aggregating over the performance + melody vectors!
  if hparams.aggregation == "sum":
    # add both mean performance and melody vectors together
    perf_mean_bias = tf.math.reduce_mean(perf_encdec_attention_bias,
                                         axis=-1, keep_dims=True)
    encoder_output = mel_encoder_output + perf_mean_vector
    encoder_decoder_attention_bias = mel_encdec_attention_bias + perf_mean_bias
  elif hparams.aggregation == "concat":
    # concatenate melody with mean-aggregated performance embedding
    stop_token = tf.zeros((1, 1, 384))
    encoder_output = tf.concat(
        [mel_encoder_output, stop_token, perf_mean_vector], axis=1)
    perf_mean_bias = tf.math.reduce_mean(perf_encdec_attention_bias,
                                         axis=-1, keep_dims=True)
    stop_bias = tf.zeros((1, 1, 1, 1))
    encoder_decoder_attention_bias = tf.concat(
        [mel_encdec_attention_bias, stop_bias, perf_mean_bias], axis=-1)
  elif hparams.aggregation == "tile":
    # tile performance embedding across each dimension of melody embedding!
    dynamic_val = tf.shape(mel_encoder_output)[1]
    shp = tf.convert_to_tensor([1, dynamic_val, 1], dtype=tf.int32)
    tiled_mean = tf.tile(perf_mean_vector, shp)

    encoder_output = tf.concat([mel_encoder_output, tiled_mean], axis=-1)
    encoder_decoder_attention_bias = mel_encdec_attention_bias
  else:
    NotImplementedError("aggregation method must be in [sum, concat, tile].")

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
        baseline=False, attention_weights=self.attention_weights,
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
      all_modalities = self._problem_hparams.modality
      return ("performance" in all_modalities) or ("inputs" in all_modalities)
    else:
      return True

  def encode(self, inputs, target_space, hparams,
             features=None, losses=None):
    """Encode transformer inputs, see transformer_encode."""
    return perf_transformer_encode(
        self._encoder_function, inputs, target_space, hparams,
        baseline=True, attention_weights=self.attention_weights,
        features=features, losses=losses,
        prepare_encoder_fn=self._prepare_encoder_fn)


@registry.register_model
class MelodyPerformanceTransformer(Transformer):
  """Learns performance embedding and concatenates it with melody embedding."""

  def __init__(self, *args, **kwargs):
    super(MelodyPerformanceTransformer, self).__init__(*args, **kwargs)
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
      all_modalities = self._problem_hparams.modality
      return ("performance" in all_modalities) or ("inputs" in all_modalities)
    else:
      return True

  def encode(self, perf_inputs, mel_inputs, target_space, hparams,
             features=None, losses=None):
    """Encode transformer inputs, but concatenate mel w perf."""
    return mel_perf_transformer_encode(
        self._encoder_function, perf_inputs, mel_inputs, target_space, hparams,
        attention_weights=self.attention_weights,
        prepare_encoder_fn=self._prepare_encoder_fn)

  def body(self, features):
    """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs. [batch_size, input_length, 1,
            hidden_dim].
          "targets": Target decoder outputs. [batch_size, decoder_length, 1,
            hidden_dim]
          "target_space_id": A scalar int from data_generators.problem.SpaceID.

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams

    losses = []

    if self.has_input:
      # extract appropriate performance and melody inputs
      perf_inputs = features["performance"]
      mel_inputs = features["melody"]
      target_space = features["target_space_id"]
      encoder_output, encoder_decoder_attention_bias = self.encode(
          perf_inputs, mel_inputs, target_space, hparams, features=features,
          losses=losses)
    else:
      encoder_output, encoder_decoder_attention_bias = (None, None)

    targets = features["targets"]
    targets_shape = common_layers.shape_list(targets)
    targets = common_layers.flatten4d3d(targets)
    decoder_input, decoder_self_attention_bias = self._prepare_decoder_fn(
        targets, hparams, features=features)

    # Not all subclasses of Transformer support keyword arguments related to
    # recurrent memory, so only pass these arguments if memory is enabled.
    decode_kwargs = {}
    if self.recurrent_memory_by_layer is not None:
      # TODO(kitaev): The chunk_number feature currently has the same shape as
      # "targets", but this is only for the purposes of sharing sharding code.
      # In fact every token within an example must have the same chunk number.
      chunk_number_each_token = tf.squeeze(features["chunk_number"], (-1, -2))
      chunk_number_each_example = chunk_number_each_token[:, 0]
      # Uncomment the code below to verify that tokens within a batch share the
      # same chunk number:
      # with tf.control_dependencies([
      #     tf.assert_equal(chunk_number_each_token,
      #                     chunk_number_each_example[:, None])
      # ]):
      #   chunk_number_each_example = tf.identity(chunk_number_each_example)
      decode_kwargs = dict(
          recurrent_memory_by_layer=self.recurrent_memory_by_layer,
          chunk_number=chunk_number_each_example,
          )
    decoder_output = self.decode(
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_self_attention_bias,
        hparams,
        nonpadding=features_to_nonpadding(features, "targets"),
        losses=losses,
        **decode_kwargs)
    expected_attentions = features.get("expected_attentions")
    if expected_attentions is not None:
      attention_loss = common_attention.encoder_decoder_attention_loss(
          expected_attentions, self.attention_weights,
          hparams.expected_attention_loss_type,
          hparams.expected_attention_loss_multiplier)
      return decoder_output, {"attention_loss": attention_loss}

    ret = tf.reshape(decoder_output, targets_shape)
    if losses:
      return ret, {"extra_loss": tf.add_n(losses)}
    else:
      return ret

  def _slow_greedy_infer(self, features, decode_length):
    """A slow greedy inference method.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": None
          "logits": `Tensor` of shape [batch_size, time, 1, 1, vocab_size].
          "losses": a dictionary: {loss-name (string): floating point `Scalar`}
      }
    """
    if not features:
      features = {}
    inputs_old = None
    # process all conditioning features
    if "inputs" in features:
      if len(features["inputs"].shape) < 4:
        inputs_old = features["inputs"]
        features["inputs"] = tf.expand_dims(features["inputs"], 2)
    else:  # this would be for melody decoding
      if "melody" in features:
        if len(features["melody"].shape) < 4:
          inputs_old = features["melody"]
          features["melody"] = tf.expand_dims(features["melody"], 2)
      if "performance" in features:
        if len(features["performance"].shape) < 4:
          inputs_old = features["performance"]
          features["performance"] = tf.expand_dims(features["performance"], 2)
    if not self.has_input:
      # Prepare partial targets.
      # In either features["inputs"] or features["targets"].
      # We force the outputs to begin with these sequences.
      partial_targets = features.get("inputs")
      if partial_targets is None:
        partial_targets = features["targets"]
      features["partial_targets"] = tf.to_int64(partial_targets)
    # Save the targets in a var and reassign it after the tf.while loop to avoid
    # having targets being in a 'while' frame. This ensures targets when used
    # in metric functions stays in the same frame as other vars.
    targets_old = features.get("targets", None)

    target_modality = self._problem_hparams.modality["targets"]

    def infer_step(recent_output, recent_logits, unused_loss):
      """Inference step."""
      if not tf.executing_eagerly():
        if self._target_modality_is_real:
          dim = self._problem_hparams.vocab_size["targets"]
          if dim is not None and hasattr(self._hparams, "vocab_divisor"):
            dim += (-dim) % self._hparams.vocab_divisor
          recent_output.set_shape([None, None, None, dim])
        else:
          recent_output.set_shape([None, None, None, 1])
      padded = tf.pad(recent_output, [[0, 0], [0, 1], [0, 0], [0, 0]])
      features["targets"] = padded
      # This is inefficient in that it generates samples at all timesteps,
      # not just the last one, except if target_modality is pointwise.
      samples, logits, losses = self.sample(features)
      # Concatenate the already-generated recent_output with last timestep
      # of the newly-generated samples.
      top = self._hparams.top.get("targets",
                                  modalities.get_top(target_modality))
      if getattr(top, "pointwise", False):
        cur_sample = samples[:, -1, :, :]
      else:
        cur_sample = samples[:,
                             common_layers.shape_list(recent_output)[1], :, :]
      if self._target_modality_is_real:
        cur_sample = tf.expand_dims(cur_sample, axis=1)
        samples = tf.concat([recent_output, cur_sample], axis=1)
      else:
        cur_sample = tf.to_int64(tf.expand_dims(cur_sample, axis=1))
        samples = tf.concat([recent_output, cur_sample], axis=1)
        if not tf.executing_eagerly():
          samples.set_shape([None, None, None, 1])

      # Assuming we have one shard for logits.
      logits = tf.concat([recent_logits, logits[:, -1:]], 1)
      loss = sum([l for l in losses.values() if l is not None])
      return samples, logits, loss

    # Create an initial output tensor. This will be passed
    # to the infer_step, which adds one timestep at every iteration.
    if "partial_targets" in features:
      initial_output = tf.to_int64(features["partial_targets"])
      while len(initial_output.get_shape().as_list()) < 4:
        initial_output = tf.expand_dims(initial_output, 2)
      batch_size = common_layers.shape_list(initial_output)[0]
    else:
      batch_size = common_layers.shape_list(features["performance"])[0]
      if self._target_modality_is_real:
        dim = self._problem_hparams.vocab_size["targets"]
        if dim is not None and hasattr(self._hparams, "vocab_divisor"):
          dim += (-dim) % self._hparams.vocab_divisor
        initial_output = tf.zeros((batch_size, 0, 1, dim), dtype=tf.float32)
      else:
        initial_output = tf.zeros((batch_size, 0, 1, 1), dtype=tf.int64)
    # Hack: foldl complains when the output shape is less specified than the
    # input shape, so we confuse it about the input shape.
    initial_output = tf.slice(initial_output, [0, 0, 0, 0],
                              common_layers.shape_list(initial_output))
    target_modality = self._problem_hparams.modality["targets"]
    if target_modality == modalities.ModalityType.CLASS_LABEL:
      decode_length = 1
    else:
      if "partial_targets" in features:
        prefix_length = common_layers.shape_list(features["partial_targets"])[1]
      else:
        # this code will generate outputs that tend to be long,
        # but this is to avoid the case when the melody is extremely short.
        # this can be changed to features["melody"] for the actual behavior.
        prefix_length = common_layers.shape_list(features["performance"])[1]
      decode_length = prefix_length + decode_length

    # Initial values of result, logits and loss.
    result = initial_output
    vocab_size = self._problem_hparams.vocab_size["targets"]
    if vocab_size is not None and hasattr(self._hparams, "vocab_divisor"):
      vocab_size += (-vocab_size) % self._hparams.vocab_divisor
    if self._target_modality_is_real:
      logits = tf.zeros((batch_size, 0, 1, vocab_size))
      logits_shape_inv = [None, None, None, None]
    else:
      # tensor of shape [batch_size, time, 1, 1, vocab_size]
      logits = tf.zeros((batch_size, 0, 1, 1, vocab_size))
      logits_shape_inv = [None, None, None, None, None]
    if not tf.executing_eagerly():
      logits.set_shape(logits_shape_inv)

    loss = 0.0

    def while_exit_cond(result, logits, loss):  # pylint: disable=unused-argument
      """Exit the loop either if reach decode_length or EOS."""
      length = common_layers.shape_list(result)[1]

      not_overflow = length < decode_length

      if self._problem_hparams.stop_at_eos:

        def fn_not_eos():
          return tf.not_equal(  # Check if the last predicted element is a EOS
              tf.squeeze(result[:, -1, :, :]), text_encoder.EOS_ID)

        not_eos = tf.cond(
            # We only check for early stopping if there is at least 1 element (
            # otherwise not_eos will crash).
            tf.not_equal(length, 0),
            fn_not_eos,
            lambda: True,
        )

        return tf.cond(
            tf.equal(batch_size, 1),
            # If batch_size == 1, we check EOS for early stopping.
            lambda: tf.logical_and(not_overflow, not_eos),
            # Else, just wait for max length
            lambda: not_overflow)
      return not_overflow

    result, logits, loss = tf.while_loop(
        while_exit_cond,
        infer_step, [result, logits, loss],
        shape_invariants=[
            tf.TensorShape([None, None, None, None]),
            tf.TensorShape(logits_shape_inv),
            tf.TensorShape([]),
        ],
        back_prop=False,
        parallel_iterations=1)
    if inputs_old is not None:  # Restore to not confuse Estimator.
      features["inputs"] = inputs_old
    # Reassign targets back to the previous value.
    if targets_old is not None:
      features["targets"] = targets_old
    losses = {"training": loss}
    if "partial_targets" in features:
      partial_target_length = common_layers.shape_list(
          features["partial_targets"])[1]
      result = tf.slice(result, [0, partial_target_length, 0, 0],
                        [-1, -1, -1, -1])
    return {
        "outputs": result,
        "scores": None,
        "logits": logits,
        "losses": losses,
    }


@registry.register_model
class BaselineMelodyTransformer(MelodyPerformanceTransformer):
  """Melody-only baseline transformer autoencoder, no mean-aggregation."""

  def __init__(self, *args, **kwargs):
    super(BaselineMelodyTransformer, self).__init__(*args, **kwargs)
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
        baseline=True, attention_weights=self.attention_weights,
        features=features, losses=losses,
        prepare_encoder_fn=self._prepare_encoder_fn)

  def body(self, features):
    """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs. [batch_size, input_length, 1,
            hidden_dim].
          "targets": Target decoder outputs. [batch_size, decoder_length, 1,
            hidden_dim]
          "target_space_id": A scalar int from data_generators.problem.SpaceID.

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams

    losses = []

    if self.has_input:
      # use melody-only as input features
      inputs = features["melody"]
      target_space = features["target_space_id"]
      encoder_output, encoder_decoder_attention_bias = self.encode(
          inputs, target_space, hparams, features=features, losses=losses)
    else:
      encoder_output, encoder_decoder_attention_bias = (None, None)

    targets = features["targets"]
    targets_shape = common_layers.shape_list(targets)
    targets = common_layers.flatten4d3d(targets)
    decoder_input, decoder_self_attention_bias = self._prepare_decoder_fn(
        targets, hparams, features=features)

    # Not all subclasses of Transformer support keyword arguments related to
    # recurrent memory, so only pass these arguments if memory is enabled.
    decode_kwargs = {}
    if self.recurrent_memory_by_layer is not None:
      # TODO(kitaev): The chunk_number feature currently has the same shape as
      # "targets", but this is only for the purposes of sharing sharding code.
      # In fact every token within an example must have the same chunk number.
      chunk_number_each_token = tf.squeeze(features["chunk_number"], (-1, -2))
      chunk_number_each_example = chunk_number_each_token[:, 0]
      # Uncomment the code below to verify that tokens within a batch share the
      # same chunk number:
      # with tf.control_dependencies([
      #     tf.assert_equal(chunk_number_each_token,
      #                     chunk_number_each_example[:, None])
      # ]):
      #   chunk_number_each_example = tf.identity(chunk_number_each_example)
      decode_kwargs = dict(
          recurrent_memory_by_layer=self.recurrent_memory_by_layer,
          chunk_number=chunk_number_each_example,
          )
    decoder_output = self.decode(
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_self_attention_bias,
        hparams,
        nonpadding=features_to_nonpadding(features, "targets"),
        losses=losses,
        **decode_kwargs)
    expected_attentions = features.get("expected_attentions")
    if expected_attentions is not None:
      attention_loss = common_attention.encoder_decoder_attention_loss(
          expected_attentions, self.attention_weights,
          hparams.expected_attention_loss_type,
          hparams.expected_attention_loss_multiplier)
      return decoder_output, {"attention_loss": attention_loss}

    ret = tf.reshape(decoder_output, targets_shape)
    if losses:
      return ret, {"extra_loss": tf.add_n(losses)}
    else:
      return ret
