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
"""Utility functions for working with files."""
import re

# Regex for matching sharded filenames of the form `<filebase>@<num shards>`.
_SHARDED_FILE_REGEX = re.compile(r'^([^@]+)@(\d+)$')


def is_sharded_filename(filename):
  """Returns True if filename is of the form `<filebase>@<num shards>`."""
  return _SHARDED_FILE_REGEX.match(filename)


def sharded_filename_to_list(sharded_filename):
  """Returns list of individual filenames based on the given sharded filename.

  A sharded filename of the form `<filebase>@<N>` is converted to the list
  [`<filebase>-00000-of-<N>`, ..., `<filebase>-<N-1>-of-<N>`]. For example,
  an input of `/tmp/examples@2` returns
  [`/tmp/examples-00000-of-00002`, `/tmp/examples-00001-of-00002`].

  Args:
    sharded_filename: A sharded filename of the form `<filebase>@<num shards>`.

  Returns:
    The list of individual files expanded from `sharded_filename`.

  Raises:
    ValueError: If not a sharded filename or the specified number of shards is
        less than 1.
  """
  if not is_sharded_filename(sharded_filename):
    raise ValueError("'%s' is not a sharded filename." % sharded_filename)

  m = _SHARDED_FILE_REGEX.search(sharded_filename)
  filebase = m.group(1)
  num_shards = int(m.group(2))
  if num_shards < 1:
    raise ValueError('At least one shard is required. Given: %d' % num_shards)

  return ['%s-%05d-of-%05d' % (filebase, i, num_shards)
          for i in range(num_shards)]


def get_filename_list(filename):
  """Returns list containing `filename` or individual shards if sharded."""
  if is_sharded_filename(filename):
    return sharded_filename_to_list(filename)
  else:
    return [filename]