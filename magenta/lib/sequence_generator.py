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
"""Abstract class for sequence generators."""

import abc

# internal imports
from magenta.protobuf import generator_pb2

class BaseSequenceGenerator(object):
  """Abstract class for generators.

  Args:
    details: A generator_pb2.GeneratorDetails for this generator.
    checkpoint_file: Where to search for the most recent model checkpoint.
  """

  __metaclass_ = abc.ABCMeta

  def __init__(self, details, checkpoint_file):
    self._details = details
    self._checkpoint_file = checkpoint_file
    self._initialized = False

  def get_details(self):
    """Returns a GeneratorDetails description of this generator."""
    return self._details

  @abc.abstractmethod
  def _initialize(self, checkpoint_file):
    """Implementation for building the TF graph.

    Must be called before _generate().
    """
    pass

  @abc.abstractmethod
  def _close(self):
    """Implementation for closing the TF session."""
    pass

  @abc.abstractmethod
  def _generate(self, generate_sequence_request):
    """Implementation for sequence generation based on request.

    Args:
      generate_sequence_request: The request for generating a sequence

    Returns:
      A GenerateSequenceResponse proto.
    """
    pass

  def initialize(self):
    """Builds the TF graph and loads the checkpoint file.

    If the graph has already been initialized, this is a no-op.

    Returns:
      A boolean indicating whether the graph was initialized.
    """
    if not self._initialized:
      self._initialize(self._checkpoint_file)
      self._initialized = True
      return True
    else:
      return False

  def close(self):
    """Closes the TF session.

    Returns:
      A boolean indicating whether the session was closed.
    """
    if self._initialized:
      self._close()
      self._initialized = False
      return True
    else:
      return False

  def __enter__(self):
    """When used as a context manager, initializes the TF session."""
    self.initialize()

  def __exit__(self):
    """When used as a context manager, closes the TF session."""
    self.close()

  def generate(self, generate_sequence_request):
    """Generates a sequence from the model based on the request.

    Also initializes the TF graph if not yet initialized.

    Args:
      generate_sequence_request: The request for generating a sequence

    Returns:
      A GenerateSequenceResponse proto.
    """
    self.initialize()
    return self._generate(generate_sequence_request)
