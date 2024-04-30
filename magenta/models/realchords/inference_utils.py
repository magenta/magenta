# Copyright 2024 The Magenta Authors.
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

"""Inference utilities for ReaLchords model generation.

The **InferenceModel is for running model inference in Colab.
"""

import functools
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from flax import linen as nn
import gin
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import seqio
from t5x import decoding
from t5x import losses as t5x_losses
from t5x import models
from t5x import optimizers
import t5x.interactive_model

Array = jnp.ndarray | np.ndarray
PyTree = Any


class MusicTransformerInferenceModel:
  """An inference model for single Transformer temprature sampling."""

  def __init__(
      self,
      model_dir: str,
      gin_file_path: str,
      step: int | None = None,
      batch_size: int = 1,
      model_parallel_submesh: Tuple[int, int, int, int] = (1, 1, 1, 1),
      lazy_init: bool = False,
      gin_overrides: str = '',
      default_decoder_params: Sequence[Tuple[str, Any]] = (
          ('temperature', 1.0),
          ('topk', 0),
      ),
      default_seed: int = 0,
      load_gin_from_model_dir: bool = True,
      additional_gin_path: Optional[List[str]] = None,
      gin_skip_unknown: bool = False,
  ):
    """Initializes the model.

    Args:
      model_dir: The directory where the model is stored.
      gin_file_path: The path to the gin file of the model.
      step: The step of the checkpoint to load.
      batch_size: The batch size at inference.
      model_parallel_submesh: The model parallel submesh.
      lazy_init: Whether to lazily initialize the model.
      gin_overrides: The gin override parameters.
      default_decoder_params: The default decoder params.
      default_seed: The default seed to be used.
      load_gin_from_model_dir: whether to load gin config from model_dir.
      additional_gin_path: Manual specification of gin config from the other
        files if there is none in model_dir. This is useful when one try to load
        a model from only the checkpoint but no gin config.
      gin_skip_unknown: Whether to skip unknown gin configs when loading gin
        files.
    """
    self._model_dir = model_dir
    self._gin_file_path = gin_file_path
    self._step = step
    self.batch_size = batch_size
    self._model_parallel_submesh = model_parallel_submesh
    self._compiled_infer_fn = None
    self._gin_overrides = gin_overrides
    self._default_decoder_params = dict(default_decoder_params)
    self._default_seed = default_seed
    self._load_gin_from_model_dir = load_gin_from_model_dir
    self._additional_gin_path = additional_gin_path
    self._gin_skip_unknown = gin_skip_unknown
    if not load_gin_from_model_dir and additional_gin_path is None:
      raise ValueError(
          'At least one of load_gin_from_model_dir and additional_gin_path'
          ' should be provided.'
      )
    if not lazy_init:
      self.initialize()

  def initialize(self):
    """Loads the model and compiles the inference function."""
    if self._compiled_infer_fn:
      return

    ckpt_path = self._model_dir
    if self._step is not None:
      ckpt_path = f'{ckpt_path}/checkpoint_{self._step}'
    gin.enter_interactive_mode()
    gin.clear_config()
    gin.parse_config_file(self._gin_file_path)
    if self._additional_gin_path is not None:
      for gin_path in self._additional_gin_path:
        gin.parse_config_file(gin_path, skip_unknown=self._gin_skip_unknown)
    if self._load_gin_from_model_dir:
      gin.parse_config_file(
          f'{self._model_dir}/config.gin', skip_unknown=self._gin_skip_unknown
      )
    gin.parse_config(self._gin_overrides)
    model = gin.get_configurable('MODEL/macro')()
    self._feature_lengths = gin.get_configurable(
        'TRAIN_TASK_FEATURE_LENGTHS/macro'
    )()

    partitioner = t5x.partitioning.PjitPartitioner(
        num_partitions=None, model_parallel_submesh=self._model_parallel_submesh
    )

    self._interactive_model = t5x.interactive_model.InteractiveModel(
        batch_size=self.batch_size,
        task_feature_lengths=self.feature_lengths,
        output_dir='/tmp',
        partitioner=partitioner,
        model=model,
        dtype=None,
        restore_mode='latest' if self._step is None else 'specific',
        checkpoint_path=ckpt_path,
        input_shapes={
            'decoder_input_tokens': (
                self.batch_size,
                self.feature_lengths['targets'],
            ),
            'decoder_causal_attention': (
                self.batch_size,
                self.feature_lengths['targets'],
            ),
        },
    )

    train_state = self._interactive_model.train_state

    def _infer_fn(params, batch, decoder_params, rng):
      assert isinstance(
          self._interactive_model.model, t5x.models.DecoderOnlyModel
      )
      return self._interactive_model.model.predict_batch_with_aux(
          params,
          batch,
          rng=rng,
          decoder_params=decoder_params,
      )

    partitioned_infer_fn = partitioner.partition(
        _infer_fn,
        # The length of this in_axis_resources need to match the length of the
        # input args of _infer_fn.
        in_axis_resources=(
            self._interactive_model.train_state_axes.params,
            partitioner.data_partition_spec,
            None,
            None,
        ),
        out_axis_resources=(
            partitioner.data_partition_spec,
            partitioner.data_partition_spec,
        ),
    )
    self._compiled_infer_fn = partitioner.compile(
        partitioned_infer_fn,
        train_state.params,
        {
            'decoder_input_tokens': jnp.zeros(
                (self.batch_size, self.feature_lengths['targets']), np.int32
            ),
            'decoder_causal_attention': jnp.ones(
                (self.batch_size, self.feature_lengths['targets']), np.int32
            ),
        },
        {
            'max_decode_steps': 2048,  # placeholder,
            **self._default_decoder_params,
        },
        random.PRNGKey(self._default_seed),  # placeholder of rng
    )
    self._initialized = True

  @property
  def feature_lengths(self) -> Mapping[str, int]:
    return self._feature_lengths

  def _infer(self, batch, decoder_params, rng):
    preds, unused_scores = self._compiled_infer_fn(
        self._interactive_model.train_state.params,
        batch,
        {**self._default_decoder_params, **decoder_params},
        rng,
    )
    return preds

  def __call__(
      self,
      decode_length: int,
      decoder_prompt: Array,
      seed: Optional[int] = None,
      decode_rng: Optional[Array] = None,
      **decoder_params: Mapping[str, Any],
  ) -> jnp.ndarray:
    """Runs inference on the model, with an optional prompt.

    Args:
      decode_length: The total length of the decoder sequence, including the
        prompt.
      decoder_prompt: An optional batch of prompts for the decoder.
      seed: The seed for random sampling.
      decode_rng: The decoder rng that will override the seed.
      **decoder_params: Additional kwargs to pass to model as `decoder_params`.

    Returns:
      The batch of predicted sequences, which excludes any supplied prompt.
    """
    self.initialize()
    prompt_length = decoder_prompt.shape[1]
    batch = {
        # Pad beginning with BOS and ending to match compilation length.
        'decoder_input_tokens': jnp.pad(
            decoder_prompt,
            [
                (0, 0),
                (1, self.feature_lengths['targets'] - prompt_length - 1),
            ],
        ),
        'decoder_causal_attention': jnp.ones(
            (self.batch_size, self.feature_lengths['targets']),
            dtype=np.int32,
        ),
    }
    decode_steps = decode_length - prompt_length
    decoder_params = {
        'max_decode_steps': jnp.array(decode_steps, dtype=jnp.int32),
        **decoder_params,
    }
    if seed is None:
      if decode_rng is None:
        rng = random.PRNGKey(self._default_seed)
      else:
        rng = decode_rng
    else:
      rng = random.PRNGKey(seed)
    preds = self._infer(batch, decoder_params, rng)

    # Unlike encoder-decoder model, the decoder-only model will remove the
    # prompt (prefix) after generation. So preds = preditions + placeholders.
    return preds[:, :decode_steps]


# Function for selecting the log prob from logits.
@jax.jit
def select_sampled(
    logits: jnp.ndarray, sampled_tokens: jnp.ndarray
) -> jnp.ndarray:
  """Select logits in an array corresponding to the sampled tokens.

  If logits=[[[.64,.2,.1], [.8,.9,.49]], and sampled_tokens=[[1, 2]
             [[42, 8, .5], [.6,.8,.21]]]                     [0, 1]]
  it returns [[.2, .49], [42, .8]].

  Args:
    logits: a batch of logits.
    sampled_tokens: a batch of sampled token ids.

  Returns:
    the sampled logits.
  """
  return jnp.squeeze(
      jnp.take_along_axis(
          logits, jnp.expand_dims(sampled_tokens, axis=-1), axis=-1
      ),
      axis=-1,
  )


class MusicTransformerEvaluateModel(MusicTransformerInferenceModel):
  """The model class for getting the log prob from the language model."""

  def __init__(
      self,
      model_dir: str,
      gin_file_path: str,
      step: int | None = None,
      batch_size: int = 1,
      model_parallel_submesh: Tuple[int, int, int, int] = (1, 1, 1, 1),
      lazy_init: bool = False,
      gin_overrides: str = '',
      default_decoder_params: Sequence[Tuple[str, Any]] = (
          ('temperature', 1.0),
          ('topk', 0),
      ),
      default_seed: int = 0,
      load_gin_from_model_dir: bool = True,
      additional_gin_path: Optional[List[str]] = None,
      gin_skip_unknown: bool = False,
  ):
    """Initializes the model.

    Args:
      model_dir: The directory where the model is stored.
      gin_file_path: The path to the gin file of the model.
      step: The step of the checkpoint to load.
      batch_size: The batch size at inference.
      model_parallel_submesh: The model parallel submesh.
      lazy_init: Whether to lazily initialize the model.
      gin_overrides: The gin override parameters.
      default_decoder_params: The default decoder params.
      default_seed: The default seed to be used.
      load_gin_from_model_dir: whether to load gin config from model_dir.
      additional_gin_path: Manual specification of gin config from the other
        files if there is none in model_dir. This is useful when one try to load
        a model from only the checkpoint but no gin config.
      gin_skip_unknown: Whether to skip unknown gin configs when loading gin
        files.
    """
    super().__init__(
        model_dir,
        gin_file_path,
        step,
        batch_size,
        model_parallel_submesh,
        lazy_init,
        gin_overrides,
        default_decoder_params,
        default_seed,
        load_gin_from_model_dir,
        additional_gin_path,
        gin_skip_unknown,
    )

  def initialize(self):
    """Loads the model and compiles the inference function."""
    if self._compiled_infer_fn:
      return

    ckpt_path = self._model_dir
    if self._step is not None:
      ckpt_path = f'{ckpt_path}/checkpoint_{self._step}'
    gin.enter_interactive_mode()
    gin.clear_config()
    gin.parse_config_file(self._gin_file_path)
    if self._additional_gin_path is not None:
      for gin_path in self._additional_gin_path:
        gin.parse_config_file(gin_path, skip_unknown=self._gin_skip_unknown)
    if self._load_gin_from_model_dir:
      gin.parse_config_file(
          f'{self._model_dir}/config.gin', skip_unknown=self._gin_skip_unknown
      )
    gin.parse_config(self._gin_overrides)
    model = gin.get_configurable('MODEL/macro')()
    self._feature_lengths = gin.get_configurable(
        'TRAIN_TASK_FEATURE_LENGTHS/macro'
    )()

    partitioner = t5x.partitioning.PjitPartitioner(
        num_partitions=None, model_parallel_submesh=self._model_parallel_submesh
    )

    self._interactive_model = t5x.interactive_model.InteractiveModel(
        batch_size=self.batch_size,
        task_feature_lengths=self.feature_lengths,
        output_dir='/tmp',
        partitioner=partitioner,
        model=model,
        dtype=None,
        restore_mode='latest' if self._step is None else 'specific',
        checkpoint_path=ckpt_path,
        input_shapes={
            'decoder_input_tokens': (
                self.batch_size,
                self.feature_lengths['targets'],
            ),
            'decoder_causal_attention': (
                self.batch_size,
                self.feature_lengths['targets'],
            ),
        },
    )

    # train_state = self._interactive_model.train_state

    def _infer_fn(params, batch):
      assert isinstance(
          self._interactive_model.model, t5x.models.DecoderOnlyModel
      )
      prompt = batch['decoder_input_tokens']
      # Add the BOS to the beginning of prompt
      prompt_shift_one = jnp.pad(prompt, [[0, 0], [1, 0]], constant_values=0)
      prompt_shift_one = prompt_shift_one[:, :-1]
      decoder_causal_attention = None  # no prefix LM in this case
      logits = self._interactive_model.model.module.apply(
          {'params': params},
          prompt_shift_one,
          prompt,
          decoder_causal_attention=decoder_causal_attention,
          decode=False,
          enable_dropout=False,
      )
      logprobs = jax.nn.log_softmax(logits, axis=-1)
      logprobs = select_sampled(logprobs, prompt)
      return logprobs

    train_state = self._interactive_model.train_state

    partitioned_infer_fn = partitioner.partition(
        _infer_fn,
        # The length of this in_axis_resources need to match the length of the
        # input args of _infer_fn.
        in_axis_resources=(
            self._interactive_model.train_state_axes.params,
            partitioner.data_partition_spec,
        ),
        out_axis_resources=partitioner.data_partition_spec,
    )
    self._compiled_infer_fn = partitioner.compile(
        partitioned_infer_fn,
        train_state.params,
        {
            'decoder_input_tokens': jnp.zeros(
                (self.batch_size, self.feature_lengths['targets']), np.int32
            ),
            'decoder_causal_attention': jnp.ones(
                (self.batch_size, self.feature_lengths['targets']), np.int32
            ),
        },
    )
    self._infer_fn = _infer_fn
    self._initialized = True

  def __call__(
      self,
      decode_length: int,
      decoder_prompt: Array,
      seed: Optional[int] = None,
      decode_rng: Optional[Array] = None,
      **decoder_params: Mapping[str, Any],
  ) -> jnp.ndarray:
    """Runs inference on the model, with an optional prompt.

    Args:
      decode_length: The total length of the decoder sequence, including the
        prompt.
      decoder_prompt: An optional batch of prompts for the decoder.
      seed: The seed for random sampling.
      decode_rng: The decoder rng that will override the seed.
      **decoder_params: Additional kwargs to pass to model as `decoder_params`.

    Returns:
      The batch of predicted sequences, which excludes any supplied prompt.
    """
    self.initialize()
    batch = {
        # Pad beginning with BOS and ending to match compilation length.
        'decoder_input_tokens': decoder_prompt,
        'decoder_causal_attention': jnp.ones(
            (self.batch_size, self.feature_lengths['targets']),
            dtype=np.int32,
        ),
    }
    logprobs = self._infer_fn(self._interactive_model.train_state.params, batch)

    return logprobs


class NonCausalMusicTransformerInferenceModel(MusicTransformerInferenceModel):
  """An inference model for non-causal encoder-decoder transformer temprature sampling."""

  def initialize(self):
    """Loads the model and compiles the inference function."""
    if self._compiled_infer_fn:
      return

    ckpt_path = self._model_dir
    if self._step is not None:
      ckpt_path = f'{ckpt_path}/checkpoint_{self._step}'
    gin.enter_interactive_mode()
    gin.clear_config()
    gin.parse_config_file(self._gin_file_path)
    if self._additional_gin_path is not None:
      for gin_path in self._additional_gin_path:
        gin.parse_config_file(gin_path, skip_unknown=self._gin_skip_unknown)
    if self._load_gin_from_model_dir:
      gin.parse_config_file(
          f'{self._model_dir}/config.gin', skip_unknown=self._gin_skip_unknown
      )
    gin.parse_config(self._gin_overrides)
    model = gin.get_configurable('MODEL/macro')()
    self._feature_lengths = gin.get_configurable(
        'TRAIN_TASK_FEATURE_LENGTHS/macro'
    )()

    partitioner = t5x.partitioning.PjitPartitioner(
        num_partitions=None, model_parallel_submesh=self._model_parallel_submesh
    )

    self._interactive_model = t5x.interactive_model.InteractiveModel(
        batch_size=self.batch_size,
        task_feature_lengths=self.feature_lengths,
        output_dir='/tmp',
        partitioner=partitioner,
        model=model,
        dtype=None,
        restore_mode='latest' if self._step is None else 'specific',
        checkpoint_path=ckpt_path,
        input_shapes={
            'encoder_input_tokens': (
                self.batch_size,
                self.feature_lengths['inputs'],
            ),
            'decoder_input_tokens': (
                self.batch_size,
                self.feature_lengths['targets'],
            ),
        },
    )

    train_state = self._interactive_model.train_state

    def _infer_fn(params, batch, decoder_params, rng):
      assert isinstance(
          self._interactive_model.model, t5x.models.EncoderDecoderModel
      )
      return self._interactive_model.model.predict_batch_with_aux(
          params,
          batch,
          rng=rng,
          prompt_with_targets=True,
          decoder_params=decoder_params,
      )

    partitioned_infer_fn = partitioner.partition(
        _infer_fn,
        # The length of this in_axis_resources need to match the length of the
        # input args of _infer_fn.
        in_axis_resources=(
            self._interactive_model.train_state_axes.params,
            partitioner.data_partition_spec,
            None,
            None,
        ),
        out_axis_resources=(
            partitioner.data_partition_spec,
            partitioner.data_partition_spec,
        ),
    )
    self._compiled_infer_fn = partitioner.compile(
        partitioned_infer_fn,
        train_state.params,
        {
            'encoder_input_tokens': jnp.zeros(
                (self.batch_size, self.feature_lengths['inputs']), np.int32
            ),
            'decoder_input_tokens': jnp.ones(
                (self.batch_size, self.feature_lengths['targets']), np.int32
            ),
        },
        {
            'max_decode_steps': 2048,  # placeholder,
            **self._default_decoder_params,
        },
        random.PRNGKey(self._default_seed),  # placeholder of rng
    )
    self._initialized = True

  def __call__(
      self,
      decode_length: int,
      decoder_prompt: Array,
      seed: Optional[int] = None,
      decode_rng: Optional[Array] = None,
      **decoder_params: Mapping[str, Any],
  ) -> jnp.ndarray:
    """Runs inference on the model, with an optional prompt.

    Args:
      decode_length: The total length of the decoder sequence, including the
        prompt.
      decoder_prompt: An optional batch of prompts for the decoder.
      seed: The seed for random sampling.
      decode_rng: The decoder rng that will override the seed.
      **decoder_params: Additional kwargs to pass to model as `decoder_params`.

    Returns:
      The batch of predicted sequences, which excludes any supplied prompt.
    """
    self.initialize()
    if 'encoder_input_tokens' not in decoder_params:
      raise ValueError('encoder_input_tokens should be provided')
    encoder_input_tokens = decoder_params.pop('encoder_input_tokens')
    prompt_length = decoder_prompt.shape[1]
    batch = {
        'encoder_input_tokens': encoder_input_tokens,
        # Pad beginning with BOS and ending to match compilation length.
        'decoder_input_tokens': jnp.pad(
            decoder_prompt,
            [
                (0, 0),
                (1, self.feature_lengths['targets'] - prompt_length - 1),
            ],
        ),
    }
    decode_steps = decode_length - prompt_length
    decoder_params = {
        'max_decode_steps': jnp.array(decode_steps, dtype=jnp.int32),
        **decoder_params,
    }
    if seed is None:
      if decode_rng is None:
        rng = random.PRNGKey(self._default_seed)
      else:
        rng = decode_rng
    else:
      rng = random.PRNGKey(seed)
    preds = self._infer(batch, decoder_params, rng)

    return preds[:, prompt_length:decode_steps]


class NonCausalMusicTransformerEvaluateModel(MusicTransformerInferenceModel):
  """An model class for getting log prob from non-causal encoder-decoder transformer temprature sampling."""

  def initialize(self):
    """Loads the model and compiles the inference function."""
    if self._compiled_infer_fn:
      return

    ckpt_path = self._model_dir
    if self._step is not None:
      ckpt_path = f'{ckpt_path}/checkpoint_{self._step}'
    gin.enter_interactive_mode()
    gin.clear_config()
    gin.parse_config_file(self._gin_file_path)
    if self._additional_gin_path is not None:
      for gin_path in self._additional_gin_path:
        gin.parse_config_file(gin_path, skip_unknown=self._gin_skip_unknown)
    if self._load_gin_from_model_dir:
      gin.parse_config_file(
          f'{self._model_dir}/config.gin', skip_unknown=self._gin_skip_unknown
      )
    gin.parse_config(self._gin_overrides)
    model = gin.get_configurable('MODEL/macro')()
    self._feature_lengths = gin.get_configurable(
        'TRAIN_TASK_FEATURE_LENGTHS/macro'
    )()

    partitioner = t5x.partitioning.PjitPartitioner(
        num_partitions=None, model_parallel_submesh=self._model_parallel_submesh
    )

    self._interactive_model = t5x.interactive_model.InteractiveModel(
        batch_size=self.batch_size,
        task_feature_lengths=self.feature_lengths,
        output_dir='/tmp',
        partitioner=partitioner,
        model=model,
        dtype=None,
        restore_mode='latest' if self._step is None else 'specific',
        checkpoint_path=ckpt_path,
        input_shapes={
            'encoder_input_tokens': (
                self.batch_size,
                self.feature_lengths['inputs'],
            ),
            'decoder_input_tokens': (
                self.batch_size,
                self.feature_lengths['targets'],
            ),
        },
    )

    train_state = self._interactive_model.train_state

    def _infer_fn(params, batch):
      assert isinstance(
          self._interactive_model.model, t5x.models.EncoderDecoderModel
      )
      decoder_input_tokens = batch['decoder_input_tokens']
      # shift by one
      decoder_input_tokens_shift_one = jnp.pad(
          decoder_input_tokens, [(0, 0), (1, 0)], constant_values=0
      )
      decoder_input_tokens_shift_one = decoder_input_tokens_shift_one[:, :-1]
      decoder_target_tokens = batch['decoder_input_tokens']
      logits = self._interactive_model.model.module.apply(
          {'params': params},
          batch['encoder_input_tokens'],
          decoder_input_tokens_shift_one,
          decoder_target_tokens,
          decode=False,
          enable_dropout=False,
      )
      logprobs = jax.nn.log_softmax(logits, axis=-1)
      logprobs = select_sampled(logprobs, decoder_target_tokens)
      return logprobs

    partitioned_infer_fn = partitioner.partition(
        _infer_fn,
        # The length of this in_axis_resources need to match the length of the
        # input args of _infer_fn.
        in_axis_resources=(
            self._interactive_model.train_state_axes.params,
            partitioner.data_partition_spec,
        ),
        out_axis_resources=partitioner.data_partition_spec,
    )
    self._compiled_infer_fn = partitioner.compile(
        partitioned_infer_fn,
        train_state.params,
        {
            'encoder_input_tokens': jnp.zeros(
                (self.batch_size, self.feature_lengths['inputs']), np.int32
            ),
            'decoder_input_tokens': jnp.ones(
                (self.batch_size, self.feature_lengths['targets']), np.int32
            ),
        },
    )
    self._infer_fn = _infer_fn
    self._initialized = True

  def __call__(
      self,
      decode_length: int,
      decoder_prompt: Array,
      seed: Optional[int] = None,
      decode_rng: Optional[Array] = None,
      **decoder_params: Mapping[str, Any],
  ) -> jnp.ndarray:
    """Runs inference on the model, with an optional prompt.

    Args:
      decode_length: The total length of the decoder sequence, including the
        prompt.
      decoder_prompt: An optional batch of prompts for the decoder.
      seed: The seed for random sampling.
      decode_rng: The decoder rng that will override the seed.
      **decoder_params: Additional kwargs to pass to model as `decoder_params`.

    Returns:
      The batch of predicted sequences, which excludes any supplied prompt.
    """
    self.initialize()
    if 'encoder_input_tokens' not in decoder_params:
      raise ValueError('encoder_input_tokens should be provided')
    encoder_input_tokens = decoder_params.pop('encoder_input_tokens')
    batch = {
        'encoder_input_tokens': encoder_input_tokens,
        # Pad beginning with BOS and ending to match compilation length.
        'decoder_input_tokens': decoder_prompt,
    }
    logprobs = self._infer_fn(self._interactive_model.train_state.params, batch)

    return logprobs


class GenWithDataDecoderOnlyModel(models.DecoderOnlyModel):
  """A model wrapper for model-data inference that conforms the DecoderOnlyModel API.

  This wrapper is a child of t5x DecoderOnlyModel which has a different
  `predict_batch_with_aux` and `_compute_logits_from_slice` method.

  During inference, in predict_batch_with_aux function, the
  batch[`decoder_target_tokens`] is added to the cache dictionary. And in
  _compute_logits_from_slice function during the sample loop, the model will
  retrieve the data from cache and do teacher-forcing and model prediction
  interleavly.
  """

  def __init__(
      self,
      module: nn.Module,
      vocabulary: seqio.Vocabulary,
      optimizer_def: optimizers.OptimizerDefType,
      decode_fn: models.DecodeFnCallable = decoding.temperature_sample,
      inputs_bidirectional_attention: bool = False,
      feature_converter_cls: Optional[
          Callable[..., seqio.FeatureConverter]
      ] = None,
      label_smoothing: float = 0.0,
      z_loss: float = 0.0,
      loss_normalizing_factor: Optional[
          Union[float, int, str, t5x_losses.SpecialLossNormalizingFactor]
      ] = None,
  ):
    super().__init__(
        module=module,
        vocabulary=vocabulary,
        optimizer_def=optimizer_def,
        decode_fn=decode_fn,
        inputs_bidirectional_attention=inputs_bidirectional_attention,
        feature_converter_cls=feature_converter_cls,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor,
    )

  def _compute_logits_from_slice(
      self,
      decoding_state: decoding.DecodingState,
      params: PyTree,
      max_decode_length: int,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Compute the logits for next step from current model.

    This function is adapted from t5x.models.
    The original function takes in previously generated token and cache,
    generate logits for the next step.

    This function is designed to sample from multiple models interleavly.

    In this function:
      1. Figure out which model should generate at current step, and get the
        corresponding parameter.
      2. Retrieve the cache corresponds to current model.
      3. Figure out for the current model, which token is generated by the
      itself
        in the previous timestep, and which token(s) is generated by the
        other model(s) in the previous timestep. Get those tokens, which we
        refer
        to prev_token and curr_token. Please see the link above for why we need
        to
        select tokens different from current sequence for each model.
      4. Update the cache for the prev_token by running one step decode call to
        model.
      5. Compute the logits for next step and cache for current step, given the
        updated cache and curr_token by running one step decode call to model.
      6. Save the cache for current model to the correspond key.

    Currently this function only supports two models. We will adapt it to
    support
      arbitrary number of models in the future.

    A more elegant solution would be write all the functions into a
    state_callback
      function, however since we need to compute the cache for the prevous
      token,
      we need to bind a DecoderOnlyModel to perform module.apply.
    So here we make a new _compute_logits_from_slice and override the original
      function after the model is created.


    Args:
      decoding_state: The decoding state that gets updated in the sample loop.
      params: The parameters of the model. We will not use this argument but
        rather select parameter of the model from decoding_state.
      max_decode_length: The maximum decoding length.

    Returns:
      flat_logits: The logits of current step sampling.
      new_flat_cache: The updated cache of current model in current step
      sampling.
    """

    def select_cache(all_cache):
      return all_cache, all_cache['decoder_0']

    def save_cache(cache_updated, all_cache):
      return {
          'decoder_target_tokens': all_cache['decoder_target_tokens'],
          'decoder_0': cache_updated,
      }

    def compute_cache_step_before(prev_token, params, flat_cache):
      _, new_vars = self.module.apply(
          {'params': params, 'cache': flat_cache},
          prev_token,
          prev_token,
          enable_dropout=False,
          decode=True,
          max_decode_length=max_decode_length,
          mutable=['cache'],
      )

      flat_cache = new_vars['cache']
      return flat_cache

    def get_logits_from_data(data, curr_index):
      """Getting fake logits that will sample deterministically the token of data."""
      # Here we take the data from the current sampling index of the data.
      # That is, we will take from the data every two steps.
      curr_data_index = curr_index
      curr_token_from_data = data[:, curr_data_index]
      vocab_size = self.module.config.vocab_size
      curr_onehot_from_data = jax.nn.one_hot(curr_token_from_data, vocab_size)
      # Replace 0 with -inf to give determinate token as sampling result.
      curr_logits_from_data = jnp.where(
          curr_onehot_from_data == 0, -jnp.inf, curr_onehot_from_data
      )
      # replace 1 with 0 to represent probability of 1.0
      curr_logits_from_data = jnp.where(
          curr_onehot_from_data == 1, 0, curr_logits_from_data
      )
      return curr_logits_from_data

    def get_logits_from_data_and_update_cache(
        decoding_state, curr_params, curr_token  # pylint: disable=unused-argument
    ):
      _, flat_cache = select_cache(decoding_state.cache)
      data = decoding_state.cache['decoder_target_tokens']
      curr_logits_from_data = get_logits_from_data(
          data, decoding_state.cur_index[0]
      )
      updated_cache = compute_cache_step_before(curr_token, params, flat_cache)
      return curr_logits_from_data, updated_cache

    def sample_from_model(decoding_state, curr_params, curr_token):
      # Select the corresponding cache from the combined cache dict.
      _, flat_cache = select_cache(decoding_state.cache)

      # Compute the next logits given the correct context and cache.
      flat_logits, new_vars = self.module.apply(
          {'params': curr_params, 'cache': flat_cache},
          curr_token,
          curr_token,
          enable_dropout=False,
          decode=True,
          max_decode_length=max_decode_length,
          mutable=['cache'],
      )

      # Remove sequence length dimension since it's always 1 during decoding.
      flat_logits = jnp.squeeze(flat_logits, axis=1)
      new_flat_cache = new_vars['cache']
      return flat_logits, new_flat_cache

    # Get current model id, parameter, number of instruments.
    curr_model_id = decoding_state.cur_index[0] % 2  # currently hack to 2
    curr_params = params
    curr_token = decoding_state.cur_token

    flat_logits, new_flat_cache = jax.lax.switch(
        curr_model_id,
        [sample_from_model, get_logits_from_data_and_update_cache],
        decoding_state,
        curr_params,
        curr_token,
    )

    # Save the cache to the corresponding key in combined cache dict.
    new_flat_cache = save_cache(new_flat_cache, decoding_state.cache)

    return flat_logits, new_flat_cache

  def predict_batch_with_aux(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      rng: Optional[jax.Array] = None,
      *,
      return_all_decodes: bool = False,
      num_decodes: int = 1,
      decoder_params: Optional[MutableMapping[str, Any]] = None,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Predict with prefix.

    `decoder_params` can be used to pass dynamic configurations to
    `self.decode_fn`. An example usage is to pass different random seed (i.e.,
    `jax.random.PRNGKey(seed)` with different `seed` value). This can be done by
    setting `decoder_params['decode_rng'] = jax.random.PRNGKey(seed)`.

    Although this method is short, there are a few subtle points that. We use a
    running example to make these points clear.

    ```
    Example
      inputs = [9, 4, 6, 1]
      targets = [3, 9, 1]

      seqio.DecoderFeatureConverter will generate these set of features

         decoder_target_tokens = [9, 4, 6, 1, 3, 9, 1, 0, 0]
          decoder_input_tokens = [0, 9, 4, 6, 1, 3, 9, 1, 0]
      decoder_causal_attention = [1, 1, 1, 1, 1, 0, 0, 0, 0]

      The output of this function is (a` through `e` are the sampled token ids):

             sampled_sequences = [9, 4, 6, 1, a, b, c, d, e].
    ```

    Given these set of features, we make a few important observation.

    1) When a decoder-only model is used for a supervised learning with "inputs"
       and "targets", one way to handle this is to concatenate the "inputs" and
       "targets". For training, we use teacher forcing for the entire
       concatenated sequence. For inference, on the other hand, we don't have
       the targets. This requires that we use teacher forcing on the "inputs"
       portion while using the generated token as the input token for the next
       decoding step. For evaluation, we do have "targets" but we only want to
       use them for computing metrics, i.e., by comparing to the sequence
       generated by the model.

       This function is currently used for evaluation mode, but by ignoring
       "targets", it can be extended for the inference mode.

    2) During evaluation mode, the targets portion is zeroed out and they are
       filled with the sampled token ids. The inputs portion is kept intact.

    3) Note that `decoder_causal_attention` has an additional 1 after the final
       "inputs" token. This is because the position where the last "inputs"
       token (in this case 1) is input and the output is the first "target"
       token (in this case 3) can be included in the non-causal attention
       region.

       This results in an alignment between `decoder_input_tokens` and
       `decoder_causal_attention` because the former is shifted to the right by
       one position. So we use `decoder_causal_attention` as a binary mask to
       zero out the target tokens in `decoder_input_tokens`.

    Note:
      In order to use a custom self._decode_fn with this model it must support:

      1) Decoding from a partially decoded state by accepting a vector of
         `initial_indices` that specify where in the input to start decoding
         from.
      2) Using a vector as the loop counter to support different examples being
         a different number of steps into their decoding loop.
      3) Be able to handle one batch element reaching `max_decode_length`
         before the others without it causing the model to prematurely stop
         decoding.

    Args:
      params: model parameters.
      batch: batch element with the model features specified in
        seqio.DecoderFeatureConverter.
      rng: an optional RNG key to use during prediction, which is passed as
        'decode_rng' to the decoding function.
      return_all_decodes: if True, will return all batch_size * num_decodes
        samples from the model as an array of shape [batch_size, num_decodes,
        sequence_length]. In this case the `num_decodes` dimension is sorted in
        increasing order of log-probability. Otherwise returns only the most
        likely samples as an array of shape [batch_size, sequence_length].
      num_decodes: number of decoded sequences to be returned.
      decoder_params: additional (model-independent) parameters for the decoder.

    Returns:
      sampled_sequences: an array of shape [batch, max_decode_length].
    """

    if 'decoder_causal_attention' not in batch:
      raise ValueError(
          'Batch does not have the right format for text generation: probably '
          'because `task_feature_lengths` passed to the feature converter does '
          'not have both `inputs` and `targets`.'
      )

    # since decoder_input_tokens is shifted to the right and
    # `decoder_causal_attention` has one more 1 than the number of inputs
    # tokens, this masks out targets portion of the decoder_input_tokens.
    inputs = batch['decoder_input_tokens'] * batch['decoder_causal_attention']

    prefilled_cache, initial_index = self._compute_kv_cache(
        params, inputs, batch['decoder_causal_attention']
    )

    cache_with_data = {
        'decoder_0': prefilled_cache,
        'decoder_target_tokens': batch['decoder_target_tokens'],
    }
    prefilled_cache = cache_with_data

    target_shape = batch['decoder_input_tokens'].shape
    max_decode_length = target_shape[1]

    tokens_ids_to_logits = functools.partial(
        self._compute_logits_from_slice,
        params=params,
        max_decode_length=max_decode_length,
    )

    if decoder_params is None:
      decoder_params = {}
    if rng is not None:
      if decoder_params.get('decode_rng') is not None:
        raise ValueError(
            f'Got RNG both from the `rng` argument ({rng}) and'
            " `decoder_params['decode_rng']`"
            f' ({decoder_params["decode_rng"]}). Please specify one or the'
            ' other.'
        )
      decoder_params['decode_rng'] = rng

    # Using the above-defined single-step decoder function, run temperature
    # sampling with the prefix.
    # [batch, max_decode_length]
    scanned = hasattr(self.module, 'scan_layers') and self.module.scan_layers

    if 'eos_id' not in decoder_params:
      decoder_params['eos_id'] = self.output_vocabulary.eos_id
    decoded_sequences, scores = self._decode_fn(
        inputs=inputs,
        cache=prefilled_cache,
        tokens_to_logits=tokens_ids_to_logits,
        num_decodes=num_decodes,
        initial_index=initial_index,
        cache_offset=1 if scanned else 0,
        **decoder_params,
    )

    if not return_all_decodes:
      # Search returns [n_batch, n_beam/decodes, n_length] with the beam/decode
      # dimension sorted in increasing order of log-probability.
      # `scores` is [batch, beam/decode_size]
      # We take the highest scoring sequence (-1) and its score
      decoded_sequences = decoded_sequences[:, -1, :]
      # Beam search returns []
      aux = {'scores': scores[:, -1]}
    else:
      # We return all samples and scores, rather than just the top ones.
      aux = {'scores': scores}

    return models.remove_prefix(decoded_sequences, initial_index), aux


class GenWithDataInferenceModel(MusicTransformerInferenceModel):
  """The inference model wrapper for GenWithDataDecoderOnlyModel.

  The only difference of this class with the base class is the input batch
    contains additional `decoder_target_tokens` which comes from `midi_tokens`
    when calling the class.
  """

  def __init__(
      self,
      model_dir: str,
      gin_file_path: str,
      step: int | None = None,
      batch_size: int = 1,
      model_parallel_submesh: Tuple[int, int, int, int] = (1, 1, 1, 1),
      lazy_init: bool = False,
      gin_overrides: str = '',
      default_decoder_params: Sequence[Tuple[str, Any]] = (
          ('temperature', 1.0),
          ('topk', 0),
      ),
      default_seed: int = 0,
      load_gin_from_model_dir: bool = True,
      additional_gin_path: Optional[List[str]] = None,
      gin_skip_unknown: bool = False,
  ):
    super().__init__(
        model_dir=model_dir,
        gin_file_path=gin_file_path,
        step=step,
        batch_size=batch_size,
        model_parallel_submesh=model_parallel_submesh,
        lazy_init=lazy_init,
        gin_overrides=gin_overrides,
        default_decoder_params=default_decoder_params,
        default_seed=default_seed,
        load_gin_from_model_dir=load_gin_from_model_dir,
        additional_gin_path=additional_gin_path,
        gin_skip_unknown=gin_skip_unknown,
    )

  def initialize(self):
    """Loads the model and compiles the inference function."""
    if self._compiled_infer_fn:
      return

    ckpt_path = self._model_dir
    if self._step is not None:
      ckpt_path = f'{ckpt_path}/checkpoint_{self._step}'
    gin.enter_interactive_mode()
    gin.clear_config()
    gin.parse_config_file(self._gin_file_path)
    if self._additional_gin_path is not None:
      for gin_path in self._additional_gin_path:
        gin.parse_config_file(gin_path, skip_unknown=self._gin_skip_unknown)
    if self._load_gin_from_model_dir:
      gin.parse_config_file(
          f'{self._model_dir}/config.gin', skip_unknown=self._gin_skip_unknown
      )
    gin.parse_config(self._gin_overrides)
    model = gin.get_configurable('MODEL/macro')()
    self._feature_lengths = gin.get_configurable(
        'TRAIN_TASK_FEATURE_LENGTHS/macro'
    )()

    partitioner = t5x.partitioning.PjitPartitioner(
        num_partitions=None, model_parallel_submesh=self._model_parallel_submesh
    )

    self._interactive_model = t5x.interactive_model.InteractiveModel(
        batch_size=self.batch_size,
        task_feature_lengths=self.feature_lengths,
        output_dir='/tmp',
        partitioner=partitioner,
        model=model,
        dtype=None,
        restore_mode='latest' if self._step is None else 'specific',
        checkpoint_path=ckpt_path,
        input_shapes={
            'decoder_input_tokens': (
                self.batch_size,
                self.feature_lengths['targets'],
            ),
            'decoder_causal_attention': (
                self.batch_size,
                self.feature_lengths['targets'],
            ),
            'decoder_target_tokens': (
                self.batch_size,
                self.feature_lengths['targets'],
            ),
        },
    )

    train_state = self._interactive_model.train_state

    def _infer_fn(params, batch, decoder_params, rng):
      assert isinstance(
          self._interactive_model.model, t5x.models.DecoderOnlyModel
      )
      return self._interactive_model.model.predict_batch_with_aux(
          params,
          batch,
          rng=rng,
          decoder_params=decoder_params,
      )

    partitioned_infer_fn = partitioner.partition(
        _infer_fn,
        in_axis_resources=(
            self._interactive_model.train_state_axes.params,
            partitioner.data_partition_spec,
            None,
            None,
        ),
        out_axis_resources=(
            partitioner.data_partition_spec,
            partitioner.data_partition_spec,
        ),
    )
    self._compiled_infer_fn = partitioner.compile(
        partitioned_infer_fn,
        train_state.params,
        {
            'decoder_input_tokens': jnp.zeros(
                (self.batch_size, self.feature_lengths['targets']), np.int32
            ),
            'decoder_causal_attention': jnp.ones(
                (self.batch_size, self.feature_lengths['targets']), np.int32
            ),
            'decoder_target_tokens': jnp.zeros(
                (self.batch_size, self.feature_lengths['targets']), np.int32
            ),
        },
        {
            'max_decode_steps': 2048,  # placeholder,
            **self._default_decoder_params,
        },
        random.PRNGKey(self._default_seed),  # placeholder of rng
    )
    self._initialized = True

  def __call__(
      self,
      decode_length: int,
      decoder_prompt: Array,
      seed: Optional[int] = None,
      decode_rng: Optional[Array] = None,
      **decoder_params: Mapping[str, Any],
  ) -> jnp.ndarray:
    """Runs inference on the model, with an optional prompt.

    Args:
      decode_length: The total length of the decoder sequence, including the
        prompt.
      decoder_prompt: An optional batch of prompts for the decoder.
      seed: The seed for random sampling.
      decode_rng: The decoder rng that will override the seed.
      **decoder_params: Additional kwargs to pass to model as `decoder_params`.

    Returns:
      The batch of predicted sequences, which excludes any supplied prompt.
    """
    self.initialize()
    prompt_length = decoder_prompt.shape[1]
    midi_tokens = decoder_params['midi_tokens']
    decoder_params.pop('midi_tokens')
    batch = {
        # Pad beginning with BOS and ending to match compilation length.
        'decoder_input_tokens': jnp.pad(
            decoder_prompt,
            [
                (0, 0),
                (1, self.feature_lengths['targets'] - prompt_length - 1),
            ],
        ),
        'decoder_causal_attention': jnp.ones(
            (self.batch_size, self.feature_lengths['targets']),
            dtype=np.int32,
        ),
        'decoder_target_tokens': midi_tokens,
    }
    decode_steps = decode_length - prompt_length
    decoder_params = {
        'max_decode_steps': jnp.array(decode_steps, dtype=jnp.int32),
        **decoder_params,
    }
    if seed is None:
      if decode_rng is None:
        rng = random.PRNGKey(self._default_seed)
      else:
        rng = decode_rng
    else:
      rng = random.PRNGKey(seed)
    preds = self._infer(batch, decoder_params, rng)

    # Unlike encoder-decoder model, the decoder-only model will remove the
    # prompt (prefix) after generation. So preds = preditions + placeholders.
    return preds[:, :decode_steps]
