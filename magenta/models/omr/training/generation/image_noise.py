"""Applies noise to an image for generating training data.

All noise assumes a monochrome image with white (255) as background.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

# internal imports
import tensorflow as tf


def placeholder_image():
  return tf.placeholder(tf.uint8, shape=(None, None), name='placeholder_image')


def random_rotation(image, angle=math.pi / 180):
  return 255. - tf.contrib.image.rotate(
      255. - tf.to_float(image),
      tf.random_uniform((), -angle, angle),
      interpolation='BILINEAR')


def gaussian_noise(image, stddev=5):
  return image + tf.random_normal(tf.shape(image), stddev=stddev)
