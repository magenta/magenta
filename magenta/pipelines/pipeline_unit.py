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
"""Defines PipelineUnit base class and implementations.

Units are data processing building blocks for creating datasets.
"""


class PipelineUnit(object):
  """Base class for data transformation modules.

  A Module is a building block for a data transformation pipeline
  (see pipeline.py). Pipeline objects run Module instances under the
  hood.

  A Module should be a self contained operation that maps an
  input type to an output type. One or many inputs and outputs are supported.
  """

  # `input_type` can be an object, a tuple of objects,
  # or a dict of name to object pairs.
  input_type = None

  # `output_type` can be an object, a tuple of objects,
  # or a dict of name to object pairs.
  output_type = None

  def __init__(self, **settings_dict):
    """Module constructor. Pass Module's settings in here."""
    pass

  def transform(self, input_object):  # pylint: disable=unused-argument
    """Run this Module's transformation from input to output.

    Args:
      input_object: An instance of `input_type`. If `input_type` is a
          tuple of objects (object_0, object_1, ...) then input will be a
          tuple of instances (object_0(), object_1(), ...). If `input_type` is
          a dict of name to object pairs
          {"name_0": object_0, "name_1": object_1, ...} then input will be a
          dict of instances {"name_0": object_0(), "name_1": object_1(), ...}.

    Returns:
      A list of instances, tuples of instances, or dicts of name to
        instance pairs depending on `output_type`. See `input` docs.
    """
    return []

  # Returns a dict of stat name to counter or histogram pairs.
  def get_stats(self):
    """Produces a dict of stats after transform is called.

    Returns:
      A dictionary of stat name to state value pairs. Stat values can be
        counters or histograms.
    """
    return {}
