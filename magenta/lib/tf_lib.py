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

  def __init__(self, **init_hparams):
    object.__setattr__(self, 'keyvals', init_hparams)

  def __getattr__(self, key):
    return self.keyvals[key]

  def __setattr__(self, key, value):
    self.keyvals[key] = value

  def parse(self, string):
    new_hparams = ast.literal_eval(string)
    return HParams(**dict(self.keyvals, **new_hparams))

  def values(self):
    return self.keyvals
