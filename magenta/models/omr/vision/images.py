"""Image utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import tensorflow as tf


# TODO(ringwalt): Replace once github.com/tensorflow/tensorflow/pull/10748
# is in.
def translate(image, x, y):
  """Translates the image.

  Args:
    image: A 2D float32 tensor.
    x: The x shift of the output, in pixels.
    y: The y shift of the output, in pixels.

  Returns:
    The translated image tensor.
  """
  # TODO(ringwalt): Fix mixing scalar constants and scalar tensors here.
  one = tf.constant(1, tf.float32)
  zero = tf.constant(0, tf.float32)
  # The inverted transformation matrix expected by tf.contrib.image.transform.
  # The last entry is the 3x3 matrix is left out and is always 1.
  translation_matrix = tf.convert_to_tensor(
      [one, zero, tf.to_float(-x),
       zero, one, tf.to_float(-y),
       zero, zero], tf.float32)  # pyformat: disable
  return tf.contrib.image.transform(image, translation_matrix)
