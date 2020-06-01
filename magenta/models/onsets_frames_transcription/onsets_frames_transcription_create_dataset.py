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

r"""Beam job for creating transcription dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.onsets_frames_transcription import configs
from magenta.models.onsets_frames_transcription import create_dataset
from magenta.models.onsets_frames_transcription import data
import tensorflow.compat.v1 as tf


def main(argv):
  del argv


  create_dataset.pipeline(
      configs.CONFIG_MAP, configs.DATASET_CONFIG_MAP, data.preprocess_example,
      data.input_tensors_to_example)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
