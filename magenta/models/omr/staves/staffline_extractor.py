"""Extracts horizontal slices from a staff for glyph classification."""
# TODO(ringwalt): Rename StafflineExtractor to PositionExtractor. Stafflines in
# this context should be renamed "extracted positions" to avoid confusion.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import enum
import tensorflow as tf

DEFAULT_TARGET_HEIGHT = 18
DEFAULT_NUM_SECTIONS = 9
DEFAULT_STAFFLINE_DISTANCE_MULTIPLE = 3


class Axes(enum.IntEnum):

  STAFF = 0
  POSITION = 1
  Y = 2
  X = 3


def get_staffline(y_position, extracted_staff_arr):
  """Gets the staffline of the extracted staff.

  Args:
    y_position: The staffline position--the relative number of notes from the
        3rd line on the staff.
    extracted_staff_arr: An extracted staff NumPy array, e.g.
        `StafflineExtractor.extract_staves()[0].eval()` (`StafflineExtractor`
        returns multiple staves).

  Returns:
    The correct staffline from `extracted_staff_arr`, with shape
        `(target_height, image_width)`.

  Raises:
    ValueError: If the `y_position` is out of bounds in either direction.
  """
  return extracted_staff_arr[y_position_to_index(y_position,
                                                 len(extracted_staff_arr))]


def y_position_to_index(y_position, num_stafflines):
  index = num_stafflines // 2 - y_position
  if not 0 <= index < num_stafflines:
    raise ValueError('y_position %d too large for %d stafflines' %
                     (y_position, num_stafflines))
  return index


class StafflineExtractor(object):
  """Extracts horizontal slices from a staff for glyph classification.

  Glyphs must be centered on either a staff line or a staff space (halfway
  between staff lines). For classification, a window is extracted with height
  2*staffline_distance around a staffline or staff space. If num_sections is 9,
  extracts the five staff lines and the staff spaces between them.

  The slice is scaled proportionally to the staffline distance, making the
  output height equal to target_height, so that the glyph classifier is
  scale-invariant.
  """

  def __init__(self, image, staves,
               target_height=DEFAULT_TARGET_HEIGHT,
               num_sections=DEFAULT_NUM_SECTIONS,
               staffline_distance_multiple=DEFAULT_STAFFLINE_DISTANCE_MULTIPLE):
    """Create the staffline extractor.

    Args:
      image: A uint8 tensor of shape (height, width). The background (usually
          white) must have a value of 0.
      staves: An instance of base.BaseStaffDetector.
      target_height: The height of the scaled output windows.
      num_sections: The number of stafflines to extract.
      staffline_distance_multiple: The height of the extracted staffline, in
          multiples of the staffline distance. For example, a notehead should
          fit in a staffline distance multiple of 1, because it starts and ends
          vertically on a staff line. However, other glyphs may need more space
          above and below to classify accurately.
    """
    self.float_image = tf.cast(image, tf.float32) / 255.
    self.staves = staves
    self.target_height = target_height
    self.num_sections = num_sections
    self.staffline_distance_multiple = staffline_distance_multiple

    # Calculate the maximum width needed.
    min_staffline_distance = tf.reduce_min(staves.staffline_distance)
    self.target_width = self._get_resized_width(min_staffline_distance)

  def extract_staves(self):
    """Extracts stafflines from all staves in the image.

    Returns:
      A float32 Tensor of shape
      (num_staves, num_sections, target_height, slice_width). If the staffline
      distance is inconsistent between staves, smaller staves will be padded
      on the right with zeros.
    """
    # Only map if we have any staves, otherwise return an empty array with the
    # correct dimensionality.
    def do_extract_staves():
      """Actually performs staffline extraction if we have any staves.

      Returns:
        The stafflines tensor. See outer function doc.
      """
      staff_ys = self.staves.staves_interpolated_y

      def extract_staff(i):
        def extract_staffline_by_index(j):
          return self._extract_staffline(
              staff_ys[i], self.staves.staffline_distance[i], j)
        return tf.map_fn(
            extract_staffline_by_index,
            tf.range(-(self.num_sections // 2), self.num_sections // 2 + 1),
            dtype=tf.float32)

      return tf.map_fn(
          extract_staff,
          tf.range(tf.shape(self.staves.staves)[0]),
          dtype=tf.float32)

    # Shape of the empty stafflines tensor, if no staves are present.
    empty_shape = (0, self.num_sections, self.target_height, 0)
    stafflines = tf.cond(
        tf.shape(self.staves.staves)[0] > 0,
        do_extract_staves,
        # Otherwise, return an empty stafflines array.
        lambda: tf.zeros(empty_shape, tf.float32))
    # We need target_height to be statically known for e.g. `util/patches.py`.
    stafflines.set_shape((None, self.num_sections, self.target_height, None))
    return stafflines

  def _extract_staffline(self, staff_y, staffline_distance, staffline_num):
    """Extracts a single staffline from a single staff."""
    # Use a float image on a 0.0-1.0 scale for classification.
    image_shape = tf.shape(self.float_image)
    height = image_shape[0]  # Can't unpack a tensor object.
    width = image_shape[1]

    # Calculate the height of the extracted staffline in the unscaled image.
    staff_window = self._get_staffline_window_size(staffline_distance)

    # Calculate the coordinates to extract for the window.
    # Note: tf.meshgrid uses xs before ys by default, but y is the 0th axis
    # for indexing.
    xs, ys = tf.meshgrid(
        tf.range(width), tf.range(staff_window) - (staff_window // 2))
    # ys are centered around 0. Add the staff_y, repeating along the
    # 0th axis.
    ys += tf.tile(staff_y[None, :], [staff_window, 1])
    # Add the offset for the staff line within the staff.
    # Round up in case the y position is not whole (in between staff lines with
    # an odd staffline distance). This puts the center of the staff space closer
    # to the center of the window.
    ys += tf.cast(
        tf.ceil(tf.truediv(staffline_num * staffline_distance, 2)), tf.int32)

    invalid = tf.logical_not(
        (0 <= ys) & (ys < height) & (0 <= xs) & (xs < width))
    # Use a coordinate of (0, 0) for pixels outside of the original image.
    # We will then fill in those pixels with zeros.
    ys = tf.where(invalid, tf.zeros_like(ys), ys)
    xs = tf.where(invalid, tf.zeros_like(xs), xs)
    inds = tf.stack([ys, xs], axis=2)
    staffline_image = tf.gather_nd(self.float_image, inds)
    # Fill the pixels outside of the original image with zeros.
    staffline_image = tf.where(
        invalid, tf.zeros_like(staffline_image), staffline_image)

    # Calculate the proportional width after scaling the height to
    # self.target_height.
    resized_width = self._get_resized_width(staffline_distance)
    # Use area resizing because we expect the output to be smaller.
    # Add extra axes, because we only have 1 image and 1 channel.
    staffline_image = tf.image.resize_area(
        staffline_image[None, :, :, None],
        [self.target_height, resized_width])[0, :, :, 0]
    # Pad to make the width consistent with target_width.
    staffline_image = tf.pad(staffline_image,
                             [[0, 0], [0, self.target_width - resized_width]])
    return staffline_image

  def _get_resized_width(self, staffline_distance):
    image_width = tf.shape(self.float_image)[1]
    window_height = self._get_staffline_window_size(staffline_distance)
    return tf.cast(
        tf.round(tf.truediv(image_width * self.target_height, window_height)),
        tf.int32)

  def _get_staffline_window_size(self, staffline_distance):
    return staffline_distance * self.staffline_distance_multiple
