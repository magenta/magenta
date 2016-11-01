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
"""Generates a stylized image given an unstylized image."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import os

# internal imports

import numpy as np
import tensorflow as tf

from magenta.models.image_stylization import image_utils
from magenta.models.image_stylization import model


flags = tf.flags
flags.DEFINE_integer('num_styles', 1,
                     'Number of styles the model was trained on.')
flags.DEFINE_string('checkpoint', None, 'Checkpoint to load the model from')
flags.DEFINE_string('input_image', None, 'Input image file')
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_string('output_basename', None, 'Output base name.')
flags.DEFINE_string('which_styles', '[0]', 'Which styles to use.')
FLAGS = flags.FLAGS


def main(unused_argv=None):
  # Load image
  image = np.expand_dims(image_utils.load_np_image(
      os.path.expanduser(FLAGS.input_image)), 0)

  which_styles = ast.literal_eval(FLAGS.which_styles)

  with tf.Graph().as_default(), tf.Session() as sess:
    stylized_images = model.transform(
        tf.concat(0, [image for _ in range(len(which_styles))]),
        normalizer_params={
            'labels': tf.constant(which_styles),
            'num_categories': FLAGS.num_styles,
            'center': True,
            'scale': True})
    model_saver = tf.train.Saver(tf.all_variables())
    checkpoint = os.path.expanduser(FLAGS.checkpoint)
    if tf.gfile.IsDirectory(checkpoint):
      checkpoint = tf.train.latest_checkpoint(checkpoint)
      tf.logging.info('loading latest checkpoint file: {}'.format(checkpoint))
    model_saver.restore(sess, checkpoint)

    output_dir = os.path.expanduser(FLAGS.output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    stylized_images = stylized_images.eval()
    for which, stylized_image in zip(which_styles, stylized_images):
      image_utils.save_np_image(
          stylized_image[None, ...],
          '{}/{}_{}.png'.format(output_dir, FLAGS.output_basename, which))


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
