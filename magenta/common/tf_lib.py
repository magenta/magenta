# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Utility functions for working with TensorFlow."""

import ast


class HParams(object):
  """Creates an object for passing around hyperparameter values.

  Use the parse method to overwrite the default hyperparameters with values
  passed in as a string representation of a Python dictionary mapping
  hyperparameters to values.

  Ex.
  hparams = magenta.common.HParams(batch_size=128, hidden_size=256)
  hparams.parse('{"hidden_size":512}')
  assert hparams.batch_size == 128
  assert hparams.hidden_size == 512
  """

  def __init__(self, **init_hparams):
    object.__setattr__(self, 'keyvals', init_hparams)

  def __getattr__(self, key):
    """Returns value of the given hyperameter, or None if does not exist."""
    return self.keyvals.get(key)

  def __setattr__(self, key, value):
    """Sets value for the hyperameter."""
    self.keyvals[key] = value

  def update(self, values_dict):
    """Merges in new hyperparameters, replacing existing with same key."""
    self.keyvals.update(values_dict)

  def parse(self, values_string):
    """Merges in new hyperparameters, replacing existing with same key."""
    self.update(ast.literal_eval(values_string))

  def values(self):
    """Return the hyperparameter values as a Python dictionary."""
    return self.keyvals
