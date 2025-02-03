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

"""Interface for interacting with non-causal model.

@Author Alex Scarlatos (scarlatos@google.com)
"""

from typing import List, Union

from absl import logging
import jax.numpy as jnp
from magenta.models.realchords import inference_utils
from magenta.models.realchords import sequence_utils
import numpy as np

NON_CAUSAL_CHORD_MODEL_PATH = 'realchords_opensource_checkpoint/offline'
PRETRAIN_MODEL_STEP = 50_000


class NonCausalModel:
  """Interface for interacting with non-causal model."""

  def __init__(self):
    self.model = inference_utils.NonCausalMusicTransformerInferenceModel(
        NON_CAUSAL_CHORD_MODEL_PATH,
        'realchords_opensource_gin/base_model_non_causal.gin',
        PRETRAIN_MODEL_STEP,
        batch_size=1,
        model_parallel_submesh=(1, 1, 1, 1),
        additional_gin_path=[
            'realchords_opensource_gin/base_model_non_causal.gin'
        ],
    )

  def get_output_tokens(
      self,
      input_tokens: Union[sequence_utils.Array, List[int]],
      temperature: float,
  ) -> List[int]:
    """Use non-causal model to generate corresponding output.

    Args:
      input_tokens: tokens from other agent to condition on
      temperature: model sampling temperature

    Returns:
      list of generated tokens, one for each frame
    """
    decode_len = len(input_tokens)
    input_tokens = jnp.expand_dims(np.array(input_tokens), axis=0)
    input_tokens = jnp.pad(
        input_tokens,
        ((0, 0), (0, 256 - input_tokens.shape[1])),
        constant_values=0,
    )
    input_tokens = sequence_utils.add_eos(input_tokens)
    logging.info('input_tokens: %s', input_tokens)
    output_tokens = self.model(
        decoder_prompt=jnp.zeros([1, 0], dtype=np.int32),
        decode_length=decode_len,
        encoder_input_tokens=input_tokens,
        seed=np.random.randint(1234),
        # Seems like type checker has bug handling kwargs type
        temperature=float(temperature),  # type: ignore
    )
    return output_tokens[0].tolist()
