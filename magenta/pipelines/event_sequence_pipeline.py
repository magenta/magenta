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

"""Pipeline for event sequences."""

from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
import tensorflow.compat.v1 as tf


class EncoderPipeline(pipeline.Pipeline):
  """A pipeline that converts an EventSequence to a model encoding."""

  def __init__(self, input_type, encoder_decoder, name=None):
    """Constructs an EncoderPipeline.

    Args:
      input_type: The type this pipeline expects as input.
      encoder_decoder: An EventSequenceEncoderDecoder.
      name: A unique pipeline name.
    """
    super(EncoderPipeline, self).__init__(
        input_type=input_type,
        output_type=tf.train.SequenceExample,
        name=name)
    self._encoder_decoder = encoder_decoder

  def transform(self, seq):
    encoded = pipelines_common.make_sequence_example(
        *self._encoder_decoder.encode(seq))
    return [encoded]
