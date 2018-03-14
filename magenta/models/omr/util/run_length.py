"""Image run length encoding.

Each run is a subsequence of consecutive pixels with the same value. The
run-length encoding is the list of all runs in order, with their lengths and
values.

See: https://en.wikipedia.org/wiki/Run-length_encoding
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import tensorflow as tf


def vertical_run_length_encoding(image):
  """Returns the runs in each column of the image.

  A run is a subsequence of consecutive pixels that all have the same value.
  Internally, we treat the image as batches of single-column images in order to
  use connected component analysis.

  Args:
    image: A 2D image.

  Returns:
    The column index of each vertical run.
    The value in the image for each vertical run.
    The length of each vertical run.
  """
  with tf.name_scope('run_length_encoding'):
    image = tf.convert_to_tensor(image, name='image', dtype=tf.bool)
    # Set arbitrary, distinct, nonzero values for True and False pixels.
    # True pixels map to 2, and False pixels map to 1.
    # Transpose the image, and insert an extra dimension. This creates a batch
    # of "images" for connected component analysis, where each image is a single
    # column of the original image. Therefore, the connected components are
    # actually runs from a single column.
    components = tf.contrib.image.connected_components(
        tf.to_int32(tf.expand_dims(tf.transpose(image), axis=1)) + 1)
    # Flatten in order to use with unsorted segment ops.
    flat_components = tf.reshape(components, [-1])

    num_components = tf.maximum(0, tf.reduce_max(components) + 1)
    # Get the column index corresponding to each pixel present in
    # flat_components.
    column_indices = tf.reshape(
        tf.tile(
            # Count 0 through `width - 1` on axis 0, then repeat each element
            # `height` times.
            tf.expand_dims(tf.range(tf.shape(image)[1]), axis=1),
            multiples=[1, tf.shape(image)[0]]),
        # pyformat: disable
        [-1])
    # Take the column index for each component. For each component index k,
    # we want any entry of column_indices where the corresponding entry in
    # flat_components is k. column_indices should be the same for all pixels in
    # the same component, so we can just take the max of all of them. Disregard
    # component 0, which just represents all of the zero pixels across the
    # entire array (should be empty, because we pass in a nonzero image).
    component_columns = tf.unsorted_segment_max(column_indices, flat_components,
                                                num_components)[1:]
    # Take the original value of each component. Again, the value should be the
    # same for all pixels in a single component, so we can just take the max.
    component_values = tf.unsorted_segment_max(
        tf.to_int32(tf.reshape(tf.transpose(image), [-1])), flat_components,
        num_components)[1:]
    # Take the length of each component (run), by counting the number of pixels
    # in the component.
    component_lengths = tf.to_int32(tf.bincount(flat_components)[1:])
    return component_columns, component_values, component_lengths
