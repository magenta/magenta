"""Staffline removal for glyph classification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import tensorflow as tf

from magenta.models.omr.util import memoize
from magenta.models.omr.util import segments

# The number of lines to remove above and below the staff center line. This
# removes the 5 staff lines, and 4 ledger lines (if present) above and below.
LINES_TO_REMOVE_ABOVE_AND_BELOW = 6


class StaffRemover(object):
  """Removes staff lines for glyph classification.

  Identifies and removes short vertical runs where we expect the staff lines.
  This means that the extracted staffline images for classification are more
  consistent, whether they are centered on the line or halfway between lines.
  """

  def __init__(self, staff_detector, threshold=127):
    self.staff_detector = staff_detector
    self.threshold = threshold

  @property
  @memoize.MemoizedFunction
  def remove_staves(self):
    """Returns the page with staff lines removed.

    Returns:
      An image of the same size as `self.staff_detector.image`, with staff lines
      erased (set to white, 255).
    """
    image = tf.convert_to_tensor(self.staff_detector.image)
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    # Max height of a run length that can be removed. Runs should have height
    # around staffline_thickness.
    max_runlength = self.staff_detector.staffline_thickness * 2

    # Calculate the expected y position of each staff line for each staff and
    # each column of the image.
    staff_center_ys = self.staff_detector.staves_interpolated_y
    all_staffline_center_ys = (
        staff_center_ys[:, None, :] +
        self.staff_detector.staffline_distance[:, None, None] * tf.range(
            -LINES_TO_REMOVE_ABOVE_AND_BELOW,
            LINES_TO_REMOVE_ABOVE_AND_BELOW + 1)[None, :, None])
    ys = tf.range(height)

    def _process_column(i):
      """Removes staves from a single column of the image.

      Args:
        i: The index of the column to remove.

      Returns:
        The single column of the image with staff lines erased.
      """
      column = image[:, i]
      # Identify runs in the column that correspond to staff lines and can be
      # erased.
      runs, run_lengths = segments.true_segments_1d(column < self.threshold)

      column_staffline_ys = all_staffline_center_ys[:, :, i]
      # The run center has to be within staffline_thickness of a staff line.
      run_matches_staffline = tf.less_equal(
          tf.reduce_min(
              tf.abs(runs[:, None, None] - column_staffline_ys[None, :, :]),
              axis=[1, 2]), self.staff_detector.staffline_thickness)

      keep_run = tf.logical_and(run_lengths < max_runlength,
                                run_matches_staffline)
      keep_run.set_shape([None])
      runs = tf.boolean_mask(runs, keep_run)
      run_lengths = tf.boolean_mask(run_lengths, keep_run)

      def do_process_column(runs, run_lengths):
        """Process the column if there are any runs matching staff lines.

        Args:
          runs: The center of each vertical run.
          run_lengths: The length of each vertical run.

        Returns:
          The column of the image with staff lines erased.
        """

        # Erase ys that belong to a run corresponding to a staff line.
        y_run_pair_distance = tf.abs(ys[:, None] - runs[None, :])
        y_runs = tf.argmin(y_run_pair_distance, axis=1)
        y_run_distance = tf.reduce_min(y_run_pair_distance, axis=1)
        y_run_lengths = tf.gather(run_lengths, y_runs)
        erase_y = tf.less_equal(y_run_distance, tf.floordiv(y_run_lengths, 2))
        white_column = tf.fill(tf.shape(column), tf.constant(255, tf.uint8))
        return tf.where(erase_y, white_column, column)

      return tf.cond(tf.shape(runs)[0] > 0,
                     lambda: do_process_column(runs, run_lengths),
                     lambda: column)

    return tf.transpose(
        tf.map_fn(
            _process_column,
            tf.range(width),
            name="staff_remover",
            dtype=tf.uint8))
