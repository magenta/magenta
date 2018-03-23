"""Staff center line filter.

Identifies candidates for the center (third line) of a staff in each column of
an image. Using the estimated staffline distance, there must be black pixels
in the expected positions of the five staff lines. Some stafflines will be
covered by a black glyph, so the black run will be thicker than the expected
staffline thickness. To account for this, at least three lines must have white
pixels both above and below them.

The filtered image is used for a Hough transform (`hough.py`) to robustly
identify staves.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import tensorflow as tf

from magenta.models.omr.util import segments

# The minimum number of columns that must have a candidate staff center line
# in a given row, to detect the row as a staff center line.
MIN_STAFF_SLICES = 0.25


def staff_center_filter(image,
                        staffline_distance,
                        staffline_thickness,
                        threshold=127):
  """Filters the image for candidate staff center lines.

  Args:
    image: The 2D tensor image.
    staffline_distance: The estimated staffline distance. Scalar tensor.
    staffline_thickness: The estimated staffline thickness. Scalar tensor.
    threshold: Scalar tensor. Pixels below the threshold are black
        (possible stafflines).

  Returns:
    A boolean tensor of the same shape as image. The candidate center staff
        lines.
  """
  image = image < threshold

  # Add is not supported for unsigned ints, so use int8 instead of uint8.
  # Dark: the image is dark where we expect a staffline.
  dark_staffline_count = tf.zeros_like(image, tf.int8)
  # Space: the image is light above and below where we expect a staffline,
  # indicating a horizontal line.
  space_staffline_count = tf.zeros_like(image, tf.int8)
  for staffline_pos in range(-2, 3):
    expected_y_line = staffline_pos * staffline_distance
    # Allow each staffline to differ slightly from the expected position.
    # The second and fourth lines can differ by 1 pixel, and the first and fifth
    # lines can differ by 2 pixels.
    # At each possible location, look for a dark pixel and light space above and
    # below.
    found_dark = tf.zeros_like(image, tf.bool)
    found_space = tf.zeros_like(image, tf.bool)
    y_adjustments = range(-abs(staffline_pos), abs(staffline_pos) + 1)
    for y_adjustment in y_adjustments:
      y_line = expected_y_line + y_adjustment
      y_above = y_line - 2 * staffline_thickness
      y_below = y_line + 2 * staffline_thickness
      found_dark |= _shift_y(image, y_line)
      found_space |= tf.logical_not(
          tf.logical_or(_shift_y(image, y_above), _shift_y(image, y_below)))
    dark_staffline_count += tf.cast(found_dark, tf.int8)
    space_staffline_count += tf.cast(found_space, tf.int8)
  return tf.logical_and(
      tf.equal(dark_staffline_count, 5),
      tf.greater_equal(space_staffline_count, 3))


def _shift_y(image, y_offset):
  """Shift the image vertically.

  Args:
    image: The 2D tensor image.
    y_offset: The vertical offset for the image.

  Returns:
    The shifted image. Each pixel is shifted up or down by y_offset. Blank space
    is filled with zeros.
  """
  height = tf.shape(image)[0]
  width = tf.shape(image)[1]

  def invalid():
    return image

  def shift_up():
    # y_offset is positive
    sliced = image[y_offset:]
    return tf.concat(
        [sliced, tf.zeros(
            [y_offset, width], dtype=image.dtype)], axis=0)

  def shift_down():
    # y_offset is negative
    sliced = image[:y_offset]
    return tf.concat(
        [tf.zeros(
            [-y_offset, width], dtype=image.dtype), sliced], axis=0)

  return tf.cond(height <= tf.abs(y_offset), invalid,
                 lambda: tf.cond(y_offset >= 0, shift_up, shift_down))


def _get_staff_ys(is_staff, staffline_thickness):
  # Return the detected staves--segments in is_staff that are roughly not much
  # bigger than staffline_thickness.
  segment_ys, segment_sizes = segments.true_segments_1d(is_staff)
  return tf.boolean_mask(segment_ys, segment_sizes <= staffline_thickness * 2)
