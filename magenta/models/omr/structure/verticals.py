"""Detects vertical lines using the runs in each column of the image.

After the TensorFlow graph is run, vertical lines are classified as stems or
barlines.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np
import tensorflow as tf

from magenta.models.omr.util import functional_ops
from magenta.models.omr.util import memoize
from magenta.models.omr.util import segments
from magenta.models.omr.vision import images
from magenta.models.omr.vision import morphology

# Join gaps in vertical lines using a small gap (relative to the difference
# between stafflines).
_DEFAULT_MAX_GAP_STAFFLINE_DISTANCE = frozenset((0.1, 0.2, 0.5))
# Beams and barlines should be at least the height of a staff (4 stafflines).
# Use a minimum of 3 * staffline distance.
_DEFAULT_MIN_LENGTH_STAFFLINE_DISTANCE = 2.5


class ColumnBasedVerticals(object):
  """Vertical line segment detector.

  Does not resolve duplicates across multiple columns, or the same line that is
  detected multiple times with different `max_gap` settings. The user should
  expect to find multiple detected lines that correspond to the same physical
  line, and choose the line that seems the most likely for a given purpose.

  Attributes:
    staff_detector: An instance of `staves.base.BaseStaffDetector`.
    image: The uint8 image tensor.
    threshold: The image threshold. int.
    thresholded_image: Whether each pixel of the image is black.
    max_gap: Multiple values for the maximum gap allowed in a line segment, in
        pixels. Tensor (1D) of ints.
    min_length: The minimum length of a line segment, in pixels. int.
  """

  def __init__(
      self,
      staff_detector,
      threshold=127,
      max_gap_staffline_distance=_DEFAULT_MAX_GAP_STAFFLINE_DISTANCE,
      min_length_staffline_distance=_DEFAULT_MIN_LENGTH_STAFFLINE_DISTANCE):
    self.staff_detector = staff_detector
    self.image = staff_detector.image
    self.threshold = threshold
    thresholded_image = tf.less(self.image, threshold)
    self.filtered_image = morphology.binary_dilation(
        _horizontal_filter(thresholded_image,
                           staff_detector.staffline_thickness), 1)
    staffline_distance = tf.reduce_mean(staff_detector.staffline_distance)
    # Deterministically convert max_gap_staffline_distance to a list.
    # We use a frozenset so there is no risk of mutating the default argument.
    self.max_gap = tf.to_int32(
        tf.round(
            tf.to_float(staffline_distance) * sorted(
                max_gap_staffline_distance)))
    self.min_length = tf.to_int32(
        tf.round(
            tf.to_float(staffline_distance) * min_length_staffline_distance))

  @property
  @memoize.MemoizedFunction
  def lines(self):
    """The vertical lines.

    Returns:
      int32 tensor of shape (num_lines, 2, 2), storing lines as
          ((start_x, start_y), (end_x, end_y)).
    """
    columns = tf.range(tf.shape(self.image)[1])
    def map_max_gap(max_gap):
      """Process all columns with the given value for max_gap."""
      return functional_ops.flat_map_fn(
          lambda column: self._verticals_in_column(max_gap, column), columns)

    return functional_ops.flat_map_fn(map_max_gap, self.max_gap)

  def _verticals_in_column(self, max_gap, column):
    """Gets the verticals from a single column.

    Args:
      max_gap: The scalar max_gap value to use. int tensor.
      column: The scalar column index. int tensor.

    Returns:
      int32 tensor of shape (num_lines_in_column, 2, 2). All start_x and end_x
          values are equal to column.
    """
    image_column = self.filtered_image[:, column]
    run_starts, run_lengths = segments.true_segments_1d(
        image_column,
        mode=segments.SegmentsMode.STARTS,
        max_gap=max_gap,
        min_length=self.min_length)
    num_runs = tf.shape(run_starts)[0]
    # x is the same for all runs in the column.
    x = tf.fill([num_runs], column)
    y0 = run_starts
    y1 = run_starts + run_lengths - 1
    return tf.stack(
        [
            tf.stack([x, y0], axis=1),
            tf.stack([x, y1], axis=1),
        ], axis=1)

  @property
  def data(self):
    """Returns the list of verticals tensors to be computed.

    Returns:
      A list of Tensors.
    """
    return [self.lines]


def _horizontal_filter(image, staffline_thickness):
  """The vertical lines horizontal filter.

  A black pixel in a vertical line must have a white pixel
  `2 * staffline_thickness` pixels away, on the left and/or right.

  Args:
    image: 2D thresholded boolean image with black pixels as True.
    staffline_thickness: The estimated staffline thickness, in pixels.

  Returns:
    The filtered image.
  """
  # images.translate() requires a float image. Unlike the convention (255 or 1.0
  # for white), the image is already thresholded here, so 1.0 is black and 0.0
  # is white.
  float_image = tf.cast(image, tf.float32)
  gap = staffline_thickness * 2
  return tf.logical_and(
      image,
      tf.logical_or(
          tf.equal(images.translate(float_image, -gap, 0), 0),
          tf.equal(images.translate(float_image, gap, 0), 0)))


class ComputedVerticals(object):
  """Computed vertical lines holder.

  The result of `ColumnBasedVerticals.compute()`. Holds a NumPy array with the
      vertical lines.
  """

  def __init__(self, lines):
    self.lines = np.array(lines)

  @property
  def data(self):
    return [self.lines]
