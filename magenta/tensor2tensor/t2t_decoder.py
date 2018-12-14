# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Tensor2Tensor decoder for Magenta problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.bin import t2t_decoder
import tensorflow as tf

# Registers all Magenta problems with Tensor2Tensor.
from magenta.tensor2tensor import problems  # pylint: disable=unused-import


def main(argv):
  t2t_decoder.main(argv)


def console_entry_point():
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
