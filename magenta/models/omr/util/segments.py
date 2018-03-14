"""Segment/run length utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import enum
import tensorflow as tf


class SegmentsMode(enum.Enum):
  """The valid modes for segmentation."""
  # Return the start position of each segment
  STARTS = 1
  # Return the floored center position of each segment
  CENTERS = 2


def true_segments_1d(segments,
                     mode=SegmentsMode.CENTERS,
                     max_gap=0,
                     min_length=0,
                     name=None):
  """Labels contiguous True runs in segments.

  Args:
    segments: 1D boolean tensor.
    mode: The SegmentsMode. Returns the start of each segment (STARTS), or the
        rounded center of each segment (CENTERS).
    max_gap: Fill gaps of length at most `max_gap` between true segments. int.
    min_length: Minimum length of a returned segment. int.
    name: Optional name for the op.

  Returns:
    run_centers: int32 tensor. Depending on `mode`, either the start of each
        True run, or the (rounded) center of each True run.
    run_lengths: int32; the lengths of each True run.
  """
  with tf.name_scope(name, "true_segments", [segments]):
    segments = tf.convert_to_tensor(segments, tf.bool)
    run_starts, run_lengths = _segments_1d(segments, mode=SegmentsMode.STARTS)
    # Take only the True runs. After whichever run is True first, the True runs
    # are every other run.
    first_run = tf.cond(
        # First value is False, or all values are False. Handles empty segments
        # correctly.
        tf.logical_or(tf.reduce_any(segments[0:1]), ~tf.reduce_any(segments)),
        lambda: tf.constant(0),
        lambda: tf.constant(1))

    num_runs = tf.shape(run_starts)[0]
    run_nums = tf.range(num_runs)
    is_true_run = tf.equal(run_nums % 2, first_run % 2)
    # Find gaps between True runs that can be merged.
    is_gap = tf.logical_and(
        tf.not_equal(run_nums % 2, first_run % 2),
        tf.logical_and(
            tf.greater(run_nums, first_run), tf.less(run_nums, num_runs - 1)))
    fill_gap = tf.logical_and(is_gap, tf.less_equal(run_lengths, max_gap))

    # Segment the consecutive runs of True or False values based on whether they
    # are True, or are a gap of False values that can be bridged. Then, flatten
    # the runs of runs.
    runs_to_merge = tf.logical_or(is_true_run, fill_gap)
    run_of_run_starts, _ = _segments_1d(runs_to_merge, mode=SegmentsMode.STARTS)

    # Get the start of every new run from the original run starts.
    merged_run_starts = tf.gather(run_starts, run_of_run_starts)
    # Make an array mapping the original runs to their run of runs. Increment
    # the number for every run of run start except for the first one, so that
    # the array has values from 0 to num_run_of_runs.
    merged_run_inds = tf.cumsum(
        tf.sparse_to_dense(
            sparse_indices=tf.cast(run_of_run_starts[1:, None], tf.int64),
            output_shape=tf.cast(num_runs[None], tf.int64),
            sparse_values=tf.ones_like(run_of_run_starts[1:])))
    # Sum the lengths of the original runs that were merged.
    merged_run_lengths = tf.segment_sum(run_lengths, merged_run_inds)

    if mode is SegmentsMode.CENTERS:
      merged_starts_or_centers = (
          merged_run_starts + tf.floordiv(merged_run_lengths - 1, 2))
    else:
      merged_starts_or_centers = merged_run_starts

    # If there are no true values, increment first_run to 1, so we will skip
    # the single (false) run.
    first_run += tf.to_int32(tf.logical_not(tf.reduce_any(segments)))

    merged_starts_or_centers = merged_starts_or_centers[first_run::2]
    merged_run_lengths = merged_run_lengths[first_run::2]

    # Only take segments at least min_length long.
    is_long_enough = tf.greater_equal(merged_run_lengths, min_length)
    is_long_enough.set_shape([None])
    merged_starts_or_centers = tf.boolean_mask(merged_starts_or_centers,
                                               is_long_enough)
    merged_run_lengths = tf.boolean_mask(merged_run_lengths, is_long_enough)

    return merged_starts_or_centers, merged_run_lengths


def _segments_1d(values, mode, name=None):
  """Labels consecutive runs of the same value.

  Args:
    values: 1D tensor of any type.
    mode: The SegmentsMode. Returns the start of each segment (STARTS), or the
        rounded center of each segment (CENTERS).
    name: Optional name for the op.

  Returns:
    run_centers: int32 tensor; the centers of each run with the same consecutive
        values.
    run_lengths: int32 tensor; the lengths of each run.

  Raises:
    ValueError: if mode is not recognized.
  """
  with tf.name_scope(name, "segments", [values]):
    def do_segments(values):
      """Actually does segmentation.

      Args:
        values: 1D tensor of any type. Non-empty.

      Returns:
        run_centers: int32 tensor
        run_lengths: int32 tensor

      Raises:
        ValueError: if mode is not recognized.
      """
      length = tf.shape(values)[0]
      values = tf.convert_to_tensor(values)
      # The first run has id 0, so we don't increment the id.
      # Otherwise, the id is incremented when the value changes.
      run_start_bool = tf.concat(
          [[False], tf.not_equal(values[1:], values[:-1])], axis=0)
      # Cumulative sum the run starts to get the run ids.
      segment_ids = tf.cumsum(tf.cast(run_start_bool, tf.int32))
      if mode is SegmentsMode.STARTS:
        run_centers = tf.segment_min(tf.range(length), segment_ids)
      elif mode is SegmentsMode.CENTERS:
        run_centers = tf.segment_mean(
            tf.cast(tf.range(length), tf.float32), segment_ids)
        run_centers = tf.cast(tf.floor(run_centers), tf.int32)
      else:
        raise ValueError("Unexpected mode: %s" % mode)
      run_lengths = tf.segment_sum(tf.ones([length], tf.int32), segment_ids)
      return run_centers, run_lengths

    def empty_segments():
      return (tf.zeros([0], tf.int32), tf.zeros([0], tf.int32))

    return tf.cond(
        tf.greater(tf.shape(values)[0], 0), lambda: do_segments(values),
        empty_segments)


def peaks(values, minval=None, invalidate_distance=0, name=None):
  """Labels peaks in values.

  Args:
    values: 1D tensor of a numeric type.
    minval: Minimum value which is considered a peak.
    invalidate_distance: Invalidates nearby potential peaks. The peaks are
        searched sequentially by descending value, and from left to right for
        equal values. Once a peak is found in this order, it invalidates any
        peaks yet to be seen that are <= invalidate_distance away. A distance of
        0 effectively produces no invalidation.
    name: Optional name for the op.

  Returns:
    peak_centers: The (rounded) centers of each peak, which are locations where
        the value is higher than the value before and after. If there is a run
        of equal values at the peak, the rounded center of the run is returned.
        int32 1D tensor.
  """
  with tf.name_scope(name, "peaks", [values]):
    values = tf.convert_to_tensor(values, name="values")
    invalidate_distance = tf.convert_to_tensor(
        invalidate_distance, name="invalidate_distance", dtype=tf.int32)
    # Segment the values and find local maxima.
    # Take the center of each run of consecutive equal values.
    segment_centers, _ = _segments_1d(values, mode=SegmentsMode.CENTERS)
    segment_values = tf.gather(values, segment_centers)
    # If we have zero or one segments, there are no peaks. Just use zeros as the
    # edge values in that case.
    first_val, second_val, penultimate_val, last_val = tf.cond(
        tf.greater_equal(tf.shape(segment_values)[0], 2),
        lambda: tuple(segment_values[i] for i in (0, 1, -2, -1)),
        lambda: tuple(tf.constant(0, values.dtype) for i in range(4)))
    # Each segment must be greater than the segment before and after it.
    segment_is_peak = tf.concat(
        [[first_val > second_val], tf.greater(
            segment_values[1:-1],
            tf.maximum(segment_values[:-2], segment_values[2:])),
         [last_val > penultimate_val]],
        axis=0)
    if minval is not None:
      # Filter the peaks by minval.
      segment_is_peak = tf.logical_and(segment_is_peak,
                                       tf.greater_equal(segment_values, minval))

    # Get the center coordinates of each peak, and sort by descending value.
    all_peaks = tf.boolean_mask(segment_centers, segment_is_peak)
    num_peaks = tf.shape(all_peaks)[0]
    peak_values = tf.boolean_mask(segment_values, segment_is_peak)
    _, peak_order = tf.nn.top_k(peak_values, k=num_peaks, sorted=True)
    all_peaks = tf.gather(all_peaks, peak_order)
    all_peaks.set_shape([None])

    # Loop over the peaks, accepting one at a time and possibly invalidating
    # other ones.
    def loop_condition(_, current_peaks):
      return tf.shape(current_peaks)[0] > 0

    def loop_body(accepted_peaks, current_peaks):
      peak = current_peaks[0]
      remaining_peaks = current_peaks[1:]

      keep_peaks = tf.greater(
          tf.abs(remaining_peaks - peak), invalidate_distance)
      remaining_peaks = tf.boolean_mask(remaining_peaks, keep_peaks)

      return tf.concat([accepted_peaks, [peak]], axis=0), remaining_peaks

    accepted_peaks = tf.while_loop(
        loop_condition,
        loop_body, [tf.zeros([0], all_peaks.dtype), all_peaks],
        shape_invariants=[tf.TensorShape([None]), tf.TensorShape([None])])[0]
    # Sort the peaks by index.
    # TODO(ringwalt): Add a tf.sort op that sorts in ascending order.
    sorted_negative_peaks, _ = tf.nn.top_k(
        -accepted_peaks, k=tf.shape(accepted_peaks)[0], sorted=True)
    return -sorted_negative_peaks
