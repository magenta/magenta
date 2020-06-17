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

"""A wrapper around the `note_seq` package for backward compatibility.

These utilities have moved to https://github.com/magenta/note_seq.
"""
import sys
import warnings

import note_seq

warnings.warn(
    '`magenta.music` is deprecated, please use the `note_seq` package in its '
    'place (https://github.com/magenta/note-seq). Importing `magenta.music` '
    'will become a failure in a future version.',
    DeprecationWarning)

sys.modules['magenta.music'] = note_seq
