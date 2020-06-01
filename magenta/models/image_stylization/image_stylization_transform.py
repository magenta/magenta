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

# Lint as: python3
"""Generates a stylized image given an unstylized image."""
import ast
import os

from magenta.models.image_stylization import image_utils
from magenta.models.image_stylization import model
from magenta.models.image_stylization import ops
import numpy as np
import tensorflow.compat.v1 as tf

flags = tf.flags
flags.DEFINE_integer('num_styles', 1,
                     'Number of styles the model was trained on.')
flags.DEFINE_float('alpha', 1.0,
                   'Width multiplier the model was trained on.')
flags.DEFINE_string('checkpoint', None, 'Checkpoint to load the model from')
flags.DEFINE_string('input_image', None, 'Input image file')
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_string('output_basename', None, 'Output base name.')
flags.DEFINE_string('which_styles', '[0]',
                    'Which styles to use. This is either a Python list or a '
                    'dictionary. If it is a list then a separate image will be '
                    'generated for each style index in the list. If it is a '
                    'dictionary which maps from style index to weight then a '
                    'single image with the linear combination of style weights '
                    'will be created. [0] is equivalent to {0: 1.0}.')
FLAGS = flags.FLAGS


def _load_checkpoint(sess, checkpoint):
  """Loads a checkpoint file into the session."""
  model_saver = tf.train.Saver(tf.global_variables())
  checkpoint = os.path.expanduser(checkpoint)
  if tf.gfile.IsDirectory(checkpoint):
    checkpoint = tf.train.latest_checkpoint(checkpoint)
    tf.logging.info('loading latest checkpoint file: {}'.format(checkpoint))
  model_saver.restore(sess, checkpoint)


def _describe_style(which_styles):
  """Returns a string describing a linear combination of styles."""
  def _format(v):
    formatted = str(int(round(v * 1000.0)))
    while len(formatted) < 3:
      formatted = '0' + formatted
    return formatted

  values = []
  for k in sorted(which_styles.keys()):
    values.append('%s_%s' % (k, _format(which_styles[k])))
  return '_'.join(values)


def _style_mixture(which_styles, num_styles):
  """Returns a 1-D array mapping style indexes to weights."""
  if not isinstance(which_styles, dict):
    raise ValueError('Style mixture must be a dictionary.')
  mixture = np.zeros([num_styles], dtype=np.float32)
  for index in which_styles:
    mixture[index] = which_styles[index]
  return mixture


def _multiple_images(input_image, which_styles, output_dir):
  """Stylizes an image into a set of styles and writes them to disk."""
  with tf.Graph().as_default(), tf.Session() as sess:
    stylized_images = model.transform(
        tf.concat([input_image for _ in range(len(which_styles))], 0),
        alpha=FLAGS.alpha,
        normalizer_params={
            'labels': tf.constant(which_styles),
            'num_categories': FLAGS.num_styles,
            'center': True,
            'scale': True
        })
    _load_checkpoint(sess, FLAGS.checkpoint)

    stylized_images = stylized_images.eval()
    for which, stylized_image in zip(which_styles, stylized_images):
      image_utils.save_np_image(
          stylized_image[None, ...],
          '{}/{}_{}.png'.format(output_dir, FLAGS.output_basename, which))


def _multiple_styles(input_image, which_styles, output_dir):
  """Stylizes image into a linear combination of styles and writes to disk."""
  with tf.Graph().as_default(), tf.Session() as sess:
    mixture = _style_mixture(which_styles, FLAGS.num_styles)
    stylized_images = model.transform(
        input_image,
        alpha=FLAGS.alpha,
        normalizer_fn=ops.weighted_instance_norm,
        normalizer_params={
            'weights': tf.constant(mixture),
            'num_categories': FLAGS.num_styles,
            'center': True,
            'scale': True
        })
    _load_checkpoint(sess, FLAGS.checkpoint)

    stylized_image = stylized_images.eval()
    image_utils.save_np_image(
        stylized_image,
        os.path.join(output_dir, '%s_%s.png' % (
            FLAGS.output_basename, _describe_style(which_styles))))


def main(unused_argv=None):
  # Load image
  image = np.expand_dims(image_utils.load_np_image(
      os.path.expanduser(FLAGS.input_image)), 0)

  output_dir = os.path.expanduser(FLAGS.output_dir)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  which_styles = ast.literal_eval(FLAGS.which_styles)
  if isinstance(which_styles, list):
    _multiple_images(image, which_styles, output_dir)
  elif isinstance(which_styles, dict):
    _multiple_styles(image, which_styles, output_dir)
  else:
    raise ValueError('--which_styles must be either a list of style indexes '
                     'or a dictionary mapping style indexes to weights.')


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
