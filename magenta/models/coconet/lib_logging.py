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

"""Utilities for structured logging of intermediate values during sampling."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os

from magenta.models.coconet import lib_util
import numpy as np


class NoLogger(object):

  def log(self, **kwargs):
    pass

  @contextlib.contextmanager
  def section(self, *args, **kwargs):
    pass


class Logger(object):
  """Keeper of structured log for intermediate values during sampling."""

  def __init__(self):
    """Initialize a Logger instance."""
    self.root = _Section("root", subsample_factor=1)
    self.stack = [self.root]

  @contextlib.contextmanager
  def section(self, label, subsample_factor=None):
    """Context manager that logs to a section nested one level deeper.

    Args:
      label: A short name for the section.
      subsample_factor: Rate at which to subsample logging in this section.

    Yields:
      yields to caller.
    """
    new_section = _Section(label, subsample_factor=subsample_factor)
    self.stack[-1].log(new_section)
    self.stack.append(new_section)
    yield
    self.stack.pop()

  def log(self, **kwargs):
    """Add a record to the log.

    Args:
      **kwargs: dictionary of key-value pairs to log.
    """
    self.stack[-1].log(kwargs)

  def dump(self, path):
    """Save the log to an npz file.

    Stores the log in structured form in an npz file. The resulting file can be
    extracted using unzip, which will write every leaf node to its own file in
    an equivalent directory structure.

    Args:
      path: the path of the npz file to which to save.
    """
    dikt = {}

    def _compile_npz_dict(item, path):
      i, node = item
      if isinstance(node, _Section):
        for subitem in node.items:
          _compile_npz_dict(subitem,
                            os.path.join(path, "%s_%s" % (i, node.label)))
      else:
        for k, v in node.items():
          dikt[os.path.join(path, "%s_%s" % (i, k))] = v

    _compile_npz_dict((0, self.root), "")
    with lib_util.atomic_file(path) as p:
      np.savez_compressed(p, **dikt)


class _Section(object):
  """A section in the Logging structure."""

  def __init__(self, label, subsample_factor=None):
    """Initialize a Section instance.

    Args:
      label: A short name for the section.
      subsample_factor: Rate at which to subsample logging in this section.
    """
    self.label = label
    self.subsample_factor = 1 if subsample_factor is None else subsample_factor
    self.items = []
    self.i = 0

  def log(self, x):
    """Add a record to the log."""
    # append or overwrite such that we retain every `subsample_factor`th value
    # and the most recent value
    item = (self.i, x)
    if (self.subsample_factor == 1 or self.i % self.subsample_factor == 1 or
        not self.items):
      self.items.append(item)
    else:
      self.items[-1] = item
    self.i += 1
