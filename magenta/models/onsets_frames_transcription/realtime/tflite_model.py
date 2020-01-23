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
"""Code for interacting with the TFLite model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf


def get_model_detail(model_path):
  interpreter = tf.lite.Interpreter(model_path=model_path)
  return (interpreter.get_input_details(), interpreter.get_output_details())


class Model(object):
  """Manage Onsets and Frames TFLite model."""

  def __init__(self, model_path):
    self._interpreter = tf.lite.Interpreter(model_path=model_path)
    self._interpreter.allocate_tensors()

  def infer(self, samples):
    """Do inference over the provided samples."""
    self._interpreter.set_tensor(
        self._interpreter.get_input_details()[0]['index'],
        samples)

    self._interpreter.invoke()
    predictions = [
        self._interpreter.get_tensor(detail['index'])
        for detail in self._interpreter.get_output_details()
    ]
    # Use axis 0 to stack but then shove the axis to the end so its
    # [time, midinote (0-88), {frame,onset, offset, velocity}]
    # The first 3 are logits, the last is simply the velocity
    result = np.transpose(np.concatenate(predictions, axis=0), [1, 2, 0])

    # TODO(fjord): Add onset filtering similar to our regular inference
    # setup.
    return result
