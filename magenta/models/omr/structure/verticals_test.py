"""Tests for vertical line detection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np
import tensorflow as tf

from magenta.models.omr.staves import testing
from magenta.models.omr.structure import verticals


class ColumnBasedVerticalsTest(tf.test.TestCase):

  def testVerticalLines_singleColumn(self):
    image = np.zeros((20, 4), bool)
    image[5:10, 0] = True
    image[11:15, 1] = True
    image[:5, 3] = True
    staff_detector = testing.FakeStaves(
        tf.constant(np.where(image, 0, 255), tf.uint8),
        staves_t=None,
        staffline_distance_t=[1],
        staffline_thickness_t=0.5)
    verticals_detector = verticals.ColumnBasedVerticals(
        staff_detector,
        max_gap_staffline_distance=[1],
        min_length_staffline_distance=3)
    lines_t = verticals_detector.lines
    with self.test_session():
      lines = [line.tolist() for line in lines_t.eval()]
    # Start is dilated by 1 pixel since the start is actually in this column.
    self.assertIn([[0, 4], [0, 14]], lines)
    # Only the end is contained in this column, so it is dilated but the start
    # is not.
    self.assertIn([[1, 5], [1, 15]], lines)
    self.assertNotIn([[2, 4], [2, 14]], lines)
    self.assertNotIn([[2, 5], [2, 15]], lines)
    self.assertIn([[3, 0], [3, 5]], lines)
    # Out of bounds.
    self.assertNotIn([[4, 0], [4, 5]], lines)


if __name__ == '__main__':
  tf.test.main()
