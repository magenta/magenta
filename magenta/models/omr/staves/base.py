"""Defines the base class for all staff detectors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

# internal imports
import numpy as np
import tensorflow as tf

from magenta.models.omr.util import memoize


class BaseStaffDetector(object):
  """Base for a routine that returns staves in a music score.

  Attributes of concrete subclasses:
    staves
    staffline_distance
    staffline_thickness
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, image=None):
    """Creates a staff detector for the given music score image.

    Args:
      image: The music score image. If none, sets self.image to a placeholder.
    """
    if image is None:
      self.image = tf.placeholder(tf.uint8, shape=(None, None))
    else:
      self.image = tf.convert_to_tensor(image, tf.uint8)

  @property
  def data(self):
    """Returns the list of staff detection tensors to be computed.

    Returns:
      A list of Tensors.
    """
    return [self.staves, self.staffline_distance, self.staffline_thickness,
            self.staves_interpolated_y]

  @property
  @memoize.MemoizedFunction
  def staves_interpolated_y(self):
    """Interpolates the center line y coordinate for each staff.

    Calculates the staff center y for each x coordinate from 0 to `width - 1`.

    Returns:
      A tensor of shape (num_staves, width).
    """
    image_shape = tf.shape(self.image)

    def _get_staff_center_line_y(staff):
      """Interpolates the y position for the staff.

      For x values in the interval [0, image_shape[1]), calculate the y
      position. The y position past either end of the staff line is assumed to
      be the same as at the endpoint.

      Args:
        staff: The sequence of (x, y) coordinates for the staff center line.
            int32 tensor of shape (num_points, 2).

      Returns:
        The array of y position values.
      """
      staff = tf.convert_to_tensor(staff, dtype=tf.int32)
      input_validation = [
          tf.Assert(
              tf.greater_equal(tf.shape(staff)[0], 2), [staff, tf.shape(staff)],
              name="at_least_2_points"),
          tf.Assert(
              tf.equal(tf.shape(staff)[1], 2), [staff, tf.shape(staff)],
              name="x_and_y"),
          tf.Assert(
              tf.greater_equal(staff[0, 0], 0), [image_shape, staff],
              name="staff_x_positive"),
          tf.Assert(
              tf.less(staff[-1, 0], image_shape[1]), [image_shape, staff],
              name="staff_x_ends_before_end_of_image"),
      ]
      # Validate the input before the main body.
      with tf.control_dependencies(input_validation):
        num_points = tf.shape(staff)[0]

      # The segments cover left of the staff, each consecutive pair of points,
      # and right of the staff.
      num_segments = num_points + 1
      def loop_body(i, ys_array):
        """Executes on each iteration of the TF while loop."""
        # Interpolate the y coordinates of the line between staff points i - 1
        # and i (i >= 1). The y coordinates correspond to x in the interval
        # [staff[i - 1, 0], staff[i, 0]).
        x0 = staff[i - 1, 0]
        y0 = staff[i - 1, 1]
        x1 = staff[i, 0]
        y1 = staff[i, 1]
        segment_ys = (tf.cast(
            tf.round(
                tf.cast(y1 - y0, tf.float32) * tf.linspace(
                    0., 1., x1 - x0 + 1)[:-1]), tf.int32) + y0)
        # Update the loop variables. Increment i, and write the current segment
        # ys to the array.
        return i + 1, ys_array.write(i, segment_ys)

      # Run a while loop to generate line segments between consecutive staff
      # points.
      all_ys_array = tf.TensorArray(
          tf.int32, infer_shape=False, size=num_segments)
      # The first segment covers [0, staff[0, 0]) (may be empty).
      all_ys_array = all_ys_array.write(
          0, tf.tile([staff[0, 1]], [staff[0, 0]]))
      # Write the segments in the interval [1, num_segments - 2].
      unused_i, all_ys_array = tf.while_loop(
          lambda i, unused_ys: i < num_segments - 1, loop_body,
          [1, all_ys_array])
      # The last segment covers [staff[-1, 0], width) (may be empty).
      all_ys_array = all_ys_array.write(
          num_segments - 1,
          tf.tile([staff[-1, 1]], [image_shape[1] - staff[-1, 0]]))
      all_ys = all_ys_array.concat()
      output_validation = [
          tf.Assert(tf.equal(tf.shape(all_ys)[0], image_shape[1]),
                    [tf.shape(all_ys), image_shape]),
      ]
      # Validate the output before returning. We need an actual op inside the
      # with statement (tf.identity).
      with tf.control_dependencies(output_validation):
        return tf.identity(all_ys)

    # The map_fn will fail if there are no staves. In that case, return an empty
    # array with the correct width.
    return tf.cond(
        tf.shape(self.staves)[0] > 0,
        lambda: tf.map_fn(_get_staff_center_line_y, self.staves),
        lambda: tf.zeros([0, image_shape[1]], tf.int32))

  def compute(self, session=None, feed_dict=None):
    """Runs staff detection.

    Args:
      session: The session to use instead of the default session.
      feed_dict: The feed dict for the TensorFlow graph.

    Returns:
      A `ComputedStaves` holding NumPy arrays for the staves.
    """
    if session is None:
      session = tf.get_default_session()
    return ComputedStaves(*session.run(self.data, feed_dict=feed_dict))


class ComputedStaves(BaseStaffDetector):
  """Computed staves holder.

  The result of `BaseStaffDetector.compute()`. Holds NumPy arrays with the
      result of staff detection.
  """

  def __init__(self, staves, staffline_distance, staffline_thickness,
               staves_interpolated_y):
    super(ComputedStaves, self).__init__()
    # TODO(ringwalt): Add a way to ensure the inputs are array-like and not
    # Tensor objects.
    self.staves = np.asarray(staves)
    self.staffline_distance = np.asarray(staffline_distance)
    self.staffline_thickness = np.asarray(staffline_thickness)
    self.staves_interpolated_y_arr = np.asarray(staves_interpolated_y)

  @property
  def staves_interpolated_y(self):
    return self.staves_interpolated_y_arr

  def compute(self, session=None, feed_dict=None):
    """Returns the already computed staves.

    Args:
      session: TensorFlow session; ignored.
      feed_dict: TensorFlow feed dict; ignored.

    Returns:
      self.
    """
    return self
