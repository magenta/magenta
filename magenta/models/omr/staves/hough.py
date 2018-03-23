"""Filtered hough staff detector.

Runs the staff center filter (see `filter.py`), and then uses the Hough
transform of the filtered image to detect nearly-horizontal lines (theta is
close to `pi / 2`).

The process is repeated for each unique detected staffline distance.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

# internal imports
import numpy as np
import tensorflow as tf

from magenta.models.omr.staves import base
from magenta.models.omr.staves import filter as staves_filter
from magenta.models.omr.staves import staffline_distance as distance
from magenta.models.omr.util import memoize
from magenta.models.omr.vision import hough

# The minimum number of columns that must have a candidate staff center line
# in a given row, to detect the row as a staff center line.
MIN_STAFF_SLICES = 0.25
DEFAULT_MAX_ABS_THETA = math.pi / 50
DEFAULT_NUM_THETA = 51


class FilteredHoughStaffDetector(base.BaseStaffDetector):
  """Filtered hough staff detector.

  Runs the staff center filter (see `filter.py`), and then uses the Hough
  transform of the filtered image to detect nearly-horizontal lines (theta is
  close to `pi / 2`).
  """

  def __init__(self, image=None, max_abs_theta=DEFAULT_MAX_ABS_THETA,
               num_theta=DEFAULT_NUM_THETA):
    """Filtered hough staff detector.

    Args:
      image: The image. If None, a placeholder will be created.
      max_abs_theta: The maximum deviation of the angle for the staff from the
          horizontal, in radians.
      num_theta: The number of thetas to be detected, between
          `pi/2 - max_abs_theta` and `pi/2 + max_abs_theta`.
    """
    super(FilteredHoughStaffDetector, self).__init__(image)
    staffline_distance, staffline_thickness = (
        distance.estimate_staffline_distance_and_thickness(self.image))
    self.estimated_staffline_distance = staffline_distance
    self.estimated_staffline_thickness = staffline_thickness
    self.max_abs_theta = float(max_abs_theta)
    self.num_theta = int(num_theta)

  @property
  def staves(self):
    staves, _ = self._data
    return staves

  @property
  def staffline_distance(self):
    _, staffline_distance = self._data
    return staffline_distance

  @property
  def staffline_thickness(self):
    return self.estimated_staffline_thickness

  @property
  @memoize.MemoizedFunction
  def _data(self):

    def detection_loop_body(i, staves, staffline_distances):
      """Per-staffline-distance staff detection loop.

      Args:
        i: The index of the current staffline distance to use.
        staves: The current staves tensor of shape (N, 2, 2).
        staffline_distances: The current staffline distance tensor. 1D with
            length N.

      Returns:
        i + 1.
        staves concatd with any newly detected staves.
        staffline_distance with the current staffline distance appended for each
            new staff.
      """
      current_staffline_distance = self.estimated_staffline_distance[i]
      current_staves = _SingleSizeFilteredHoughStaffDetector(
          self.image, current_staffline_distance,
          self.estimated_staffline_thickness, self.max_abs_theta,
          self.num_theta).staves
      staves = tf.concat([staves, current_staves], axis=0)
      staffline_distances = tf.concat(
          [
              staffline_distances,
              tf.tile([current_staffline_distance], tf.shape(staves)[0:1]),
          ],
          axis=0)
      return i + 1, staves, staffline_distances

    num_staffline_distances = tf.shape(self.estimated_staffline_distance)[0]
    _, staves, staffline_distances = tf.while_loop(
        lambda i, _, __: tf.less(i, num_staffline_distances),
        detection_loop_body, [
            tf.constant(0),
            tf.zeros([0, 2, 2], tf.int32),
            tf.zeros([0], tf.int32)
        ],
        shape_invariants=[
            tf.TensorShape(()),
            tf.TensorShape([None, 2, 2]),
            tf.TensorShape([None])
        ],
        parallel_iterations=1)

    # Sort by y0.
    order, = _argsort(staves[:, 0, 1])
    staves = tf.gather(staves, order)
    staffline_distances = tf.gather(staffline_distances, order)

    return staves, staffline_distances


class _SingleSizeFilteredHoughStaffDetector(object):
  """Filtered hough staff detector for a single staffline distance size.

  This is run in a loop by `FilteredHoughStaffDetector` in order to cover all of
  the detected staffline distances.

  Runs the staff center filter (see `filter.py`), and then uses the Hough
  transform of the filtered image to detect nearly-horizontal lines (theta is
  close to `pi / 2`).
  """

  def __init__(self, image, staffline_distance, staffline_thickness,
               max_abs_theta, num_theta):
    """Filtered hough staff detector.

    Args:
      image: The image. If None, a placeholder will be created.
      staffline_distance: The single staffline distance scalar to use.
      staffline_thickness: The staffline thickness.
      max_abs_theta: The maximum deviation of the angle for the staff from the
          horizontal, in radians.
      num_theta: The number of thetas to be detected, between
          `pi/2 - max_abs_theta` and `pi/2 + max_abs_theta`.
    """
    self.image = image
    self.estimated_staffline_distance = staffline_distance
    self.estimated_staffline_thickness = staffline_thickness
    self.max_abs_theta = float(max_abs_theta)
    self.num_theta = int(num_theta)

  # Memoize this to not re-compute "staves" as a different tensor each time this
  # property is referenced. TF's common subexpression elimination doesn't seem
  # to handle this case, maybe because we have too many ops.
  @property
  def staves(self):
    """The staves detected for a single staffline distance.

    Returns:
      A staves tensor of shape (N, 2, 2).
    """
    height = tf.shape(self.image)[0]
    width = tf.shape(self.image)[1]
    staff_center = staves_filter.staff_center_filter(
        self.image, self.estimated_staffline_distance,
        self.estimated_staffline_thickness)
    all_thetas = tf.linspace(math.pi / 2 - self.max_abs_theta,
                             math.pi / 2 + self.max_abs_theta, self.num_theta)
    hough_bins = hough.hough_lines(staff_center, all_thetas)
    staff_rhos, staff_thetas = hough.hough_peaks(
        hough_bins,
        all_thetas,
        minval=MIN_STAFF_SLICES * tf.cast(width, tf.float32),
        invalidate_distance=self.estimated_staffline_distance * 4)
    num_staves = tf.shape(staff_rhos)[0]
    # Interpolate the start and end points for the staff center line.
    x0 = tf.zeros([num_staves], tf.int32)
    y0 = tf.cast(
        tf.cast(staff_rhos, tf.float32) / tf.sin(staff_thetas), tf.int32)
    x1 = tf.fill([num_staves], width - 1)
    y1 = tf.cast((tf.cast(staff_rhos, tf.float32) - tf.cast(
        width - 1, tf.float32) * tf.cos(staff_thetas)) / tf.sin(staff_thetas),
                 tf.int32)
    # Cut out staves which have a start or end y outside of the image.
    is_valid = tf.logical_and(
        tf.logical_and(0 <= y0, y0 < height),
        tf.logical_and(0 <= y1, y1 < height))
    staves = tf.reshape(tf.stack([x0, y0, x1, y1], axis=1), [-1, 2, 2])
    return tf.boolean_mask(staves, is_valid)


# TODO(ringwalt): Add tf.argsort.
def _argsort(values):
  return tf.py_func(np.argsort, [values], [tf.int64])
