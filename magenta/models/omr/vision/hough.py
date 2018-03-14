"""Hough transform for line detection.

Transforms a boolean image to the Hough space, where each entry corresponds to a
line parameterized by the angle `theta` clockwise from vertical (in radians),
and the distance `rho` (in pixels; the distance from coordinate `(0, 0)` in the
image to the closest point in the line).

For performance, the image should be sparse, containing mostly False elements,
because `tf.where(image)` will be called.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import tensorflow as tf

from magenta.models.omr.util import segments


def hough_lines(image, thetas):
  """Hough transform of a boolean image.

  Args:
    image: The image. 2D boolean tensor. Should be sparse (mostly Falses).
    thetas: 1D float32 tensor of possible angles from the vertical for the line.

  Returns:
    The Hough space for the image. Shape `(num_theta, num_rho)`, where `num_rho`
        is `sqrt(height**2 + width**2)`.
  """
  coords = tf.cast(tf.where(image), thetas.dtype)
  rho = tf.cast(
      # x cos theta + y sin theta
      tf.expand_dims(coords[:, 1], 0) * tf.cos(thetas)[:, None] +
      tf.expand_dims(coords[:, 0], 0) * tf.sin(thetas)[:, None],
      tf.int32)

  height = tf.cast(tf.shape(image)[0], tf.float64)
  width = tf.cast(tf.shape(image)[1], tf.float64)
  num_rho = tf.cast(tf.ceil(tf.sqrt(height * height + width * width)), tf.int32)
  hough_bins = _bincount_2d(rho, num_rho)
  return hough_bins


def hough_peaks(hough_bins, thetas, minval=0, invalidate_distance=0):
  """Finds the peak lines in Hough space.

  Args:
    hough_bins: Hough bins returned by `hough_lines`.
    thetas: Angles; argument given to `hough_lines`.
    minval: Minimum vote count for a Hough bin to be considered. int or float.
    invalidate_distance: When selecting a line `(rho, theta)`, invalidate all
        lines with the same theta and `+- invalidate_distance` from `rho`.
        int32. Caveat: this should only be used if all theta values are similar.
        If thetas cover a wide range, this will invalidate lines that might not
        even intersect.

  Returns:
    Tensor of peak rho indices (int32).
    Tensor of peak theta values (float32).
  """
  thetas = tf.convert_to_tensor(thetas)
  bin_score_dtype = thetas.dtype  # floating point score derived from hough_bins
  minval = tf.convert_to_tensor(minval)
  if minval.dtype.is_floating:
    minval = tf.ceil(minval)
  invalidate_distance = tf.convert_to_tensor(
      invalidate_distance, dtype=tf.int32)
  # Choose the theta with the highest bin value for each rho.
  selected_theta_ind = tf.argmax(hough_bins, axis=0)
  # Take the Hough bin value for each rho and the selected theta.
  hough_bins = tf.gather_nd(hough_bins,
                            tf.stack(
                                [
                                    tf.cast(selected_theta_ind, tf.int32),
                                    tf.range(tf.shape(hough_bins)[1])
                                ],
                                axis=1))
  # hough_bins are integers. Subtract a penalty (< 1) for lines that are not
  # horizontal or vertical, so that we break ties in favor of the more
  # horizontal or vertical line.
  infinitesimal = tf.constant(1e-10, bin_score_dtype)
  # Decrease minval so we don't discard bins that are penalized, if they
  # originally equalled minval.
  minval = tf.cast(minval, bin_score_dtype) - infinitesimal
  selected_thetas = tf.gather(thetas, selected_theta_ind)
  # min(|sin(t)|, |cos(t)|) is 0 for horizontal and vertical angles, and between
  # 0 and 1 otherwise.
  penalty = tf.multiply(
      tf.minimum(
          tf.abs(tf.sin(selected_thetas)), tf.abs(tf.cos(selected_thetas))),
      infinitesimal)
  bin_score = tf.cast(hough_bins, bin_score_dtype) - penalty
  # Find the peaks in the 1D hough_bins array.
  peak_rhos = segments.peaks(
      bin_score, minval=minval, invalidate_distance=invalidate_distance)
  # Get the actual angles for each selected peak.
  peak_thetas = tf.gather(thetas, tf.gather(selected_theta_ind, peak_rhos))
  return peak_rhos, peak_thetas


def _bincount_2d(values, num_values):
  """Bincounts each row of values.

  Args:
    values: The values to bincount. 2D integer tensor.
    num_values: The number of columns of the output. Entries in `values` that
        are `>= num_values` will be ignored.

  Returns:
    The bin counts. Shape `(values.shape[0], num_values)`. The `i`th row
        contains the result of
        `tf.bincount(values[i, :], maxlength=num_values)`.
  """
  num_rows = tf.shape(values)[0]
  # Convert the values in each row to a consecutive range of ids that will not
  # overlap with the other rows.
  row_values = values + tf.range(num_rows)[:, None] * num_values
  # Remove entries that would collide with other rows.
  values_flat = tf.boolean_mask(row_values,
                                (0 <= values) & (values < num_values))
  bins_length = num_rows * num_values
  bins = tf.bincount(values_flat, minlength=bins_length, maxlength=bins_length)
  return tf.reshape(bins, [num_rows, num_values])
