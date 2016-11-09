r"""Code to extract a tensorflow checkpoint from a bundle file.

To run this code on your local machine:
$ bazel run magenta/scripts:unpack_bundle -- \
--bundle_path 'path' --checkpoint_path 'path'
"""

import os

import tensorflow as tf

from magenta.music import sequence_generator_bundle

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
