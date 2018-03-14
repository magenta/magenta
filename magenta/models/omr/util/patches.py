"""Extracts patches from image(s)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import six
import tensorflow as tf


def patches_1d(images, patch_width):
  """Extract patches along the last dimension of `images`.

  Thin wrapper around `tf.extract_image_patches` that only takes horizontal
  slices of the input, and reshapes the output to N-D images.

  Args:
    images: The image(s) to extract patches from. Shape at least 2D, with shape
        images_shape + (height, width). The image height and width must be
        statically known (set on `images.get_shape()`).
    patch_width: Width of a patch. Python integer (must be statically known).

  Returns:
    Patches extracted from each image in `images`. Shape
        images_shape + (width - patch_width + 1, height, width).

  Raises:
    ValueError: If patch_width is not an integer.
  """
  if not isinstance(patch_width, six.integer_types):
    raise ValueError("patch_width must be an integer")
  # The shape of the input, excluding image height and width.
  images_shape = tf.shape(images)[:-2]
  # num_images is not necessary to know at graph creation time.
  num_images = tf.reduce_prod(images_shape)
  # image_height must be statically known for tf.extract_image_patches.
  image_height = int(images.get_shape()[-2])
  image_width = tf.shape(images)[-1]

  def do_extract_patches():
    """Returns the image patches, assuming images.shape[0] > 0."""
    # patch_width must be an int, not a Tensor.
    # Reshape to (num_images, height, width, channels).
    images_nhwc = tf.reshape(images,
                             tf.stack(
                                 [num_images, image_height, image_width, 1]))
    patches = tf.extract_image_patches(
        images_nhwc, [1, image_height, patch_width, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding="VALID")
    patches_shape = tf.concat(
        [images_shape, [tf.shape(patches)[2], image_height, patch_width]],
        axis=0)
    return tf.reshape(patches, patches_shape)

  def empty_patches():
    return tf.zeros(
        tf.concat([images_shape, [0, image_height, patch_width]], axis=0),
        images.dtype)

  return tf.cond(
      tf.greater(num_images, 0), do_extract_patches, empty_patches)
