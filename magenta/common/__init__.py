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

"""Imports objects into the top-level common namespace."""

from __future__ import absolute_import

from .beam_search import beam_search
from .nade import Nade
from .sequence_example_lib import count_records
from .sequence_example_lib import flatten_maybe_padded_sequences
from .sequence_example_lib import get_padded_batch
from .tf_utils import merge_hparams
