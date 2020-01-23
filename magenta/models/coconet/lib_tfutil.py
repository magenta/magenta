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

"""Utilities that depend on Tensorflow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf


# adapts batch size in response to ResourceExhaustedErrors
class RobustPredictor(object):
  """A wrapper for predictor functions that automatically adapts batch size.

  Predictor functions are functions that take batched model inputs and return
  batched model outputs. RobustPredictor partitions the batch in response to
  ResourceExhaustedErrors, making multiple calls to the wrapped predictor to
  process the entire batch.
  """

  def __init__(self, predictor):
    """Initialize a RobustPredictor instance."""
    self.predictor = predictor
    self.maxsize = None

  def __call__(self, pianoroll, mask):
    """Call the wrapped predictor and return its output."""
    if self.maxsize is not None and pianoroll.size > self.maxsize:
      return self._bisect(pianoroll, mask)
    try:
      return self.predictor(pianoroll, mask)
    except tf.errors.ResourceExhaustedError:
      if self.maxsize is None:
        self.maxsize = pianoroll.size
      self.maxsize = int(self.maxsize / 2)
      print('ResourceExhaustedError on batch of %s elements, lowering max size '
            'to %s' % (pianoroll.size, self.maxsize))
      return self._bisect(pianoroll, mask)

  def _bisect(self, pianoroll, mask):
    i = int(len(pianoroll) / 2)
    if i == 0:
      raise ValueError('Batch size is zero!')
    return np.concatenate(
        [self(pianoroll[:i], mask[:i]),
         self(pianoroll[i:], mask[i:])], axis=0)


class WrappedModel(object):
  """A data structure that holds model, graph and hparams."""

  def __init__(self, model, graph, hparams):
    self.model = model
    self.graph = graph
    self.hparams = hparams
