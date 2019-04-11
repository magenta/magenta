# Copyright 2019 The Magenta Authors.
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

r"""Code to extract a tensorflow checkpoint from a bundle file.

To run this code on your local machine:
$ python magenta/scripts/unpack_bundle.py \
--bundle_path 'path' --checkpoint_path 'path'
"""

from magenta.music import sequence_generator_bundle
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('bundle_path', '',
                           'Path to .mag file containing the bundle')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/model.ckpt',
                           'Path where the extracted checkpoint should'
                           'be saved')


def main(_):
  bundle_file = FLAGS.bundle_path
  checkpoint_file = FLAGS.checkpoint_path
  metagraph_filename = checkpoint_file + '.meta'

  bundle = sequence_generator_bundle.read_bundle_file(bundle_file)

  with tf.gfile.Open(checkpoint_file, 'wb') as f:
    f.write(bundle.checkpoint_file[0])

  with tf.gfile.Open(metagraph_filename, 'wb') as f:
    f.write(bundle.metagraph_file)

if __name__ == '__main__':
  tf.app.run()
