"""Binary morphology ops.

See: https://en.wikipedia.org/wiki/Mathematical_morphology#Binary_morphology

From the link above, these functions use a structuring element of `Z^2`--the
neighbors of a pixel are the pixels above, below, left, and right, which are
assumed to be False if they lie outside the image.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import tensorflow as tf

from magenta.models.omr.vision import images


def binary_erosion(image, n):
  """The binary erosion of a boolean image.

  True pixels that border a False pixel will be set to False.

  Args:
    image: 2D boolean tensor.
    n: Integer scalar tensor. Repeat the erosion `n` times.

  Returns:
    The eroded image.
  """
  with tf.name_scope("binary_erosion"):
    image = tf.convert_to_tensor(image, tf.bool, "image")
    result = _repeated_morphological_op(tf.to_float(image), tf.logical_and, n)
    return tf.cast(result, tf.bool)


def binary_dilation(image, n):
  """The binary dilation of a boolean image.

  False pixels that border a True pixel will be set to True.

  Args:
    image: 2D boolean tensor.
    n: Integer scalar tensor. Repeat the dilation `n` times.

  Returns:
    The dilated image.
  """
  with tf.name_scope("binary_dilation"):
    image = tf.convert_to_tensor(image, tf.bool, "image")
    result = _repeated_morphological_op(tf.to_float(image), tf.logical_or, n)
    return tf.cast(result, tf.bool)


def _repeated_morphological_op(float_image, binary_op, n):

  def body(i, image):
    return i + 1, _single_morphological_op(image, binary_op)

  return tf.while_loop(lambda i, _: tf.less(i, n), body,
                       [tf.constant(0), float_image])[1]


def _single_morphological_op(float_image, binary_op):
  with tf.name_scope("_single_morphological_op"):
    input_image = float_image
    for x, y in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
      float_image = tf.to_float(
          binary_op(
              tf.cast(float_image, tf.bool),
              tf.cast(images.translate(input_image, x, y), tf.bool)))
    return float_image
