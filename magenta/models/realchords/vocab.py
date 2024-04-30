# Copyright 2024 The Magenta Authors.
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

"""Vocabulary for ReaLchords."""

from magenta.models.realchords import event_codec
import seqio

_PAD_ID = 0
_EOS_ID = 1


def get_vocabulary(codec: event_codec.Codec) -> seqio.Vocabulary:
  return seqio.PassThroughVocabulary(size=codec.num_classes, eos_id=_EOS_ID)


def get_vocab_size(vocabulary: seqio.Vocabulary) -> int:
  return vocabulary.vocab_size
