"""Implements staffline distance estimation.

The staffline distance is the vertical distance between consecutive lines in a
staff, which is assumed to be uniform for a single staff on a scanned music
score. The staffline thickness is the vertical height of each staff line, which
is assumed to be uniform for the entire page.

Uses the algorithm described in [1], which creates a histogram of possible
staffline distance and thickness values for the entire image, based on the
vertical run-length encoding [2]. Each consecutive pair of black and white runs
contributes to the staffline distance histogram (because they may be the
staffline followed by an unobstructed space, or vice versa). We then take the
argmax of the histogram, and find candidate staff line runs. These runs must be
before or after another run, such that the sum of the run lengths is the
detected staffline distance. Then the black run is considered to be an actual
staff line, and its length contributes to the staffline thickness histogram.

Although we use a single staffline distance value for staffline thickness
detection, we may detect multiple distinct peaks in the histogram. We then run
staff detection using each distinct peak value, to detect smaller staves with an
unusual size, e.g. ossia parts [3].

[1] Cardoso, Jaime S., and Ana Rebelo. "Robust staffline thickness and distance
    estimation in binary and gray-level music scores." 20th International
    Conference on Pattern Recognition (ICPR). IEEE, 2010.
[2] https://en.wikipedia.org/wiki/Run-length_encoding
[3] https://en.wikipedia.org/wiki/Ossia
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import tensorflow as tf

from magenta.models.omr.util import run_length
from magenta.models.omr.util import segments

# The size of the histograms. Normal values for the peak are around 20 for
# staffline distance, and 2-3 for staffline thickness.
_MAX_STAFFLINE_DISTANCE_THICKNESS_VALUE = 256

# The minimum number of votes for a staffline distance bin. We expect images to
# be a reasonable size (> 100x100), and want to ensure we exclude images that
# don't contain any staves.
_MIN_STAFFLINE_DISTANCE_SCORE = 10000

# The maximum allowed number of unique staffline distances. If more staffline
# distances are detected, return an empty list instead.
_MAX_ALLOWED_UNIQUE_STAFFLINE_DISTANCES = 3

_STAFFLINE_DISTANCE_INVALIDATE_DISTANCE = 1
_STAFFLINE_THICKNESS_INVALIDATE_DISTANCE = 1
_PEAK_CUTOFF = 0.5


def _single_peak(values, relative_cutoff, minval, invalidate_distance):
  """Takes a single peak if it is high enough compared to all other peaks.

  Args:
    values: 1D tensor of values to take the peaks on.
    relative_cutoff: The fraction of the highest peak which all other peaks
        should be below.
    minval: The peak should have at least this value.
    invalidate_distance: Exclude values that are up to invalidate_distance away
        from the peak.

  Returns:
    The index of the single peak in `values`, or -1 if there is not a single
        peak that satisfies `relative_cutoff`.
  """
  relative_cutoff = tf.convert_to_tensor(relative_cutoff, tf.float32)

  # argmax is safe because the histogram is always non-empty.
  peak = tf.to_int32(tf.argmax(values))
  # Take values > minval away from the peak.
  other_values = tf.boolean_mask(
      values,
      tf.greater(
          tf.abs(tf.range(tf.shape(values)[0]) - peak), invalidate_distance))
  should_take_peak = tf.logical_and(
      tf.greater_equal(values[peak], minval),
      # values[peak] * relative_cutoff must be >= other_values.
      tf.reduce_all(
          tf.greater_equal(
              tf.to_float(values[peak]) * relative_cutoff,
              tf.to_float(other_values))))
  return tf.cond(should_take_peak, lambda: peak, lambda: -1)


def _estimate_staffline_distance(columns, lengths):
  """Estimates the staffline distances of a music score.

  Args:
    columns: 1D array. The column indices of each vertical run.
    lengths: 1D array. The length of each consecutive vertical run.

  Returns:
    A 1D tensor of possible staffline distances in the image.
  """
  with tf.name_scope('estimate_staffline_distance'):
    run_pair_lengths = lengths[:-1] + lengths[1:]
    keep_pair = tf.equal(columns[:-1], columns[1:])
    staffline_distance_histogram = tf.bincount(
        tf.boolean_mask(run_pair_lengths, keep_pair),
        # minlength required to avoid errors on a fully white image.
        minlength=_MAX_STAFFLINE_DISTANCE_THICKNESS_VALUE,
        maxlength=_MAX_STAFFLINE_DISTANCE_THICKNESS_VALUE)
    peaks = segments.peaks(
        staffline_distance_histogram,
        minval=_MIN_STAFFLINE_DISTANCE_SCORE,
        invalidate_distance=_STAFFLINE_DISTANCE_INVALIDATE_DISTANCE)

    def do_filter_peaks():
      """Process the peaks if they are non-empty.

      Returns:
        The filtered peaks. Peaks below the cutoff when compared to the highest
            peak are removed. If the peaks are invalid, then an empty list is
            returned.
      """
      histogram_size = tf.shape(staffline_distance_histogram)[0]
      peak_values = tf.to_float(tf.gather(staffline_distance_histogram, peaks))
      max_value = tf.reduce_max(peak_values)
      allowed_peaks = tf.greater_equal(peak_values,
                                       max_value * tf.constant(_PEAK_CUTOFF))

      # Check if there are too many detected staffline distances, and we should
      # return an empty list.
      allowed_peaks &= tf.less_equal(
          tf.reduce_sum(tf.to_int32(allowed_peaks)),
          _MAX_ALLOWED_UNIQUE_STAFFLINE_DISTANCES)

      # Check if any values sufficiently far away from the peaks are too high.
      # This means the peaks are not sharp enough and we should return an empty
      # list.
      far_from_peak = tf.greater(
          tf.reduce_min(
              tf.abs(tf.range(histogram_size)[None, :] - peaks[:, None]),
              axis=0), _STAFFLINE_DISTANCE_INVALIDATE_DISTANCE)
      allowed_peaks &= tf.less(
          tf.to_float(
              tf.reduce_max(
                  tf.boolean_mask(staffline_distance_histogram,
                                  far_from_peak))),
          max_value * tf.constant(_PEAK_CUTOFF))

      return tf.boolean_mask(peaks, allowed_peaks)

    return tf.cond(
        tf.greater(tf.shape(peaks)[0], 0), do_filter_peaks,
        lambda: tf.identity(peaks))


def _estimate_staffline_thickness(columns, values, lengths, staffline_distance):
  """Estimates the staffline thickness of a music score.

  Args:
    columns: 1D array. The column indices of each consecutive vertical run.
    values: 1D array. The value (0 or 1) of each vertical run.
    lengths: 1D array. The length of each vertical run.
    staffline_distance: A 1D tensor of the possible staffline distances in the
        image. One of the distances may be chosen arbitrarily.

  Returns:
    A scalar tensor with the staffline thickness for the entire page, or -1 if
      it could not be estimated (staffline_distance is empty, or there are not
      enough runs to estimate the staffline thickness).
  """

  with tf.name_scope('estimate_staffline_thickness'):

    def do_estimate():
      """Compute the thickness if distance detection was successful."""
      run_pair_lengths = lengths[:-1] + lengths[1:]
      # Use the smallest staffline distance to estimate the staffline thickness.
      keep_pair = tf.logical_and(
          tf.equal(columns[:-1], columns[1:]),
          tf.equal(run_pair_lengths, staffline_distance[0]))

      run_pair_lengths = tf.boolean_mask(run_pair_lengths, keep_pair)
      start_values = tf.boolean_mask(values[:-1], keep_pair)
      start_lengths = tf.boolean_mask(lengths[:-1], keep_pair)
      end_lengths = tf.boolean_mask(lengths[1:], keep_pair)

      staffline_thickness_values = tf.where(
          tf.not_equal(start_values, 0), start_lengths, end_lengths)
      staffline_thickness_histogram = tf.bincount(
          staffline_thickness_values,
          minlength=_MAX_STAFFLINE_DISTANCE_THICKNESS_VALUE,
          maxlength=_MAX_STAFFLINE_DISTANCE_THICKNESS_VALUE)

      return _single_peak(
          staffline_thickness_histogram,
          _PEAK_CUTOFF,
          minval=1,
          invalidate_distance=_STAFFLINE_THICKNESS_INVALIDATE_DISTANCE)

    return tf.cond(
        tf.greater(tf.shape(staffline_distance)[0], 0), do_estimate,
        lambda: tf.constant(-1, tf.int32))


def estimate_staffline_distance_and_thickness(image, threshold=127):
  """Estimates the staffline distance and thickness of a music score.

  Args:
    image: A 2D tensor (HW) and type uint8.
    threshold: The global threshold for the image.

  Returns:
    The estimated vertical distance(s) from the center of one staffline to the
        next in the music score. 1D tensor containing all unique values of the
        estimated staffline distance for each staff.
    The estimated staffline thickness of the music score.

  Raises:
    TypeError: If `image` is an invalid type.
  """
  image = tf.convert_to_tensor(image, name='image', dtype=tf.uint8)
  threshold = tf.convert_to_tensor(threshold, name='threshold', dtype=tf.uint8)
  if image.dtype.base_dtype != tf.uint8:
    raise TypeError('Invalid dtype %s.' % image.dtype)

  columns, values, lengths = run_length.vertical_run_length_encoding(
      tf.less(image, threshold))

  staffline_distance = _estimate_staffline_distance(columns, lengths)
  staffline_thickness = _estimate_staffline_thickness(columns, values, lengths,
                                                      staffline_distance)
  # staffline_thickness may be -1 even if staffline_distance > 0. Fix it so
  # that we can check either one to determine whether there are staves.
  staffline_distance = tf.cond(
      tf.equal(staffline_thickness, -1), lambda: tf.zeros([0], tf.int32),
      lambda: tf.identity(staffline_distance))
  return staffline_distance, staffline_thickness
