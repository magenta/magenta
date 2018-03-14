"""Tests for the segmentation utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
from absl.testing import absltest
import numpy as np
import tensorflow as tf

from magenta.models.omr.util import segments


class SegmentsTest(tf.test.TestCase, absltest.TestCase):

  def test_true_segments_1d(self):
    # Arbitrary boolean array to get True and False runs from.
    values = tf.constant([True, True, True, False, True, False, True, True])
    centers, lengths = segments.true_segments_1d(
        values, mode=segments.SegmentsMode.CENTERS)
    with self.test_session():
      self.assertAllEqual(centers.eval(), [1, 4, 6])
      self.assertAllEqual(lengths.eval(), [3, 1, 2])
    starts, lengths = segments.true_segments_1d(
        values, mode=segments.SegmentsMode.STARTS)
    with self.test_session():
      self.assertAllEqual(starts.eval(), [0, 4, 6])
      self.assertAllEqual(lengths.eval(), [3, 1, 2])

  def test_true_segments_1d_large(self):
    # Arbitrary boolean array to get True and False runs from.
    run_values = [False, True, False, True, False, True, False, True]
    run_lengths = [3, 5, 2, 6, 4, 8, 7, 1]
    values = tf.constant(np.repeat(run_values, run_lengths))
    centers, lengths = segments.true_segments_1d(
        values, mode=segments.SegmentsMode.CENTERS)
    with self.test_session():
      self.assertAllEqual(
          centers.eval(),
          [sum(run_lengths[:1]) + (run_lengths[1] - 1) // 2,
           sum(run_lengths[:3]) + (run_lengths[3] - 1) // 2,
           sum(run_lengths[:5]) + (run_lengths[5] - 1) // 2,
           sum(run_lengths[:7]) + (run_lengths[7] - 1) // 2])
      self.assertAllEqual(lengths.eval(), run_lengths[1::2])
    starts, lengths = segments.true_segments_1d(
        values, mode=segments.SegmentsMode.STARTS)
    with self.test_session():
      self.assertAllEqual(
          starts.eval(),
          [sum(run_lengths[:1]), sum(run_lengths[:3]),
           sum(run_lengths[:5]), sum(run_lengths[:7])])
      self.assertAllEqual(lengths.eval(), run_lengths[1::2])

  def test_true_segments_1d_empty(self):
    for mode in list(segments.SegmentsMode):
      for max_gap in [0, 1]:
        centers, lengths = segments.true_segments_1d(
            [], mode=mode, max_gap=max_gap)
        with self.test_session():
          self.assertAllEqual(centers.eval(), [])
          self.assertAllEqual(lengths.eval(), [])

  def test_true_segments_1d_max_gap(self):
    # Arbitrary boolean array to get True and False runs from.
    values = tf.constant([
        False, False,
        True, True, True,
        False, False,
        True,
        False, False, False, False, False, False,
        True, True, True, True,
        False,
        True, True,
        False, False,
        True,
    ])  # pyformat: disable
    centers, lengths = segments.true_segments_1d(values, max_gap=0)
    with self.test_session():
      self.assertAllEqual(centers.eval(), [3, 7, 15, 19, 23])
      self.assertAllEqual(lengths.eval(), [3, 1, 4, 2, 1])
    centers, lengths = segments.true_segments_1d(values, max_gap=1)
    with self.test_session():
      self.assertAllEqual(centers.eval(), [3, 7, 17, 23])
      self.assertAllEqual(lengths.eval(), [3, 1, 7, 1])
    for max_gap in range(2, 6):
      centers, lengths = segments.true_segments_1d(values, max_gap=max_gap)
      with self.test_session():
        self.assertAllEqual(centers.eval(), [4, 18])
        self.assertAllEqual(lengths.eval(), [6, 10])
    centers, lengths = segments.true_segments_1d(values, max_gap=6)
    with self.test_session():
      self.assertAllEqual(centers.eval(), [12])
      self.assertAllEqual(lengths.eval(), [22])

  # TODO(ringwalt): Make these tests parameterized when absl is released.
  def test_true_segments_1d_all_false_length_1(self):
    self._test_true_segments_1d_all_false(1)

  def test_true_segments_1d_all_false_length_2(self):
    self._test_true_segments_1d_all_false(2)

  def test_true_segments_1d_all_false_length_8(self):
    self._test_true_segments_1d_all_false(8)

  def test_true_segments_1d_all_false_length_11(self):
    self._test_true_segments_1d_all_false(11)

  def _test_true_segments_1d_all_false(self, length):
    centers, lengths = segments.true_segments_1d(tf.zeros(length, tf.bool))
    with self.test_session():
      self.assertAllEqual(centers.eval(), [])
      self.assertAllEqual(lengths.eval(), [])

  def test_true_segments_1d_min_length_0(self):
    self._test_true_segments_1d_min_length(0)

  def test_true_segments_1d_min_length_1(self):
    self._test_true_segments_1d_min_length(1)

  def test_true_segments_1d_min_length_2(self):
    self._test_true_segments_1d_min_length(2)

  def test_true_segments_1d_min_length_3(self):
    self._test_true_segments_1d_min_length(3)

  def test_true_segments_1d_min_length_4(self):
    self._test_true_segments_1d_min_length(4)

  def test_true_segments_1d_min_length_5(self):
    self._test_true_segments_1d_min_length(5)

  def test_true_segments_1d_min_length_6(self):
    self._test_true_segments_1d_min_length(6)

  def _test_true_segments_1d_min_length(self, min_length):
    # Arbitrary boolean array to get True and False runs from.
    values = tf.constant([
        False, False, False,
        True,
        False,
        True, True,
        False,
        True,
        False,
        True, True, True, True,
        False,
        True, True,
    ])  # pyformat: disable
    all_centers = np.asarray([3, 5, 8, 11, 15])
    all_lengths = np.asarray([1, 2, 1, 4, 2])
    expected_centers = all_centers[all_lengths >= min_length]
    expected_lengths = all_lengths[all_lengths >= min_length]
    centers, lengths = segments.true_segments_1d(values, min_length=min_length)
    with self.test_session():
      self.assertAllEqual(expected_centers, centers.eval())
      self.assertAllEqual(expected_lengths, lengths.eval())

  def test_peaks(self):
    values = tf.constant([5, 3, 1, 1, 0, 1, 2, 3, 3, 3, 2, 3, 4, 1, 2])
    with self.test_session():
      self.assertAllEqual(segments.peaks(values).eval(), [0, 8, 12, 14])
      self.assertAllEqual(segments.peaks(values, minval=3).eval(), [0, 8, 12])

  def test_peaks_empty(self):
    with self.test_session():
      self.assertAllEqual(segments.peaks([]).eval(), [])

  def test_peaks_invalidate_distance(self):
    values = tf.constant([0, 0, 10, 0, 5, 3, 2, 1, 2, 3, 8, 8, 7, 8])
    with self.test_session():
      self.assertAllEqual(
          segments.peaks(values, invalidate_distance=0).eval(), [2, 4, 10, 13])
      self.assertAllEqual(
          segments.peaks(values, invalidate_distance=1).eval(), [2, 4, 10, 13])
      self.assertAllEqual(
          segments.peaks(values, invalidate_distance=2).eval(), [2, 10, 13])
      self.assertAllEqual(
          segments.peaks(values, invalidate_distance=3).eval(), [2, 10])
      self.assertAllEqual(
          segments.peaks(values, invalidate_distance=4).eval(), [2, 10])
      self.assertAllEqual(
          segments.peaks(values, invalidate_distance=7).eval(), [2, 10])
      self.assertAllEqual(
          segments.peaks(values, invalidate_distance=8).eval(), [2, 13])
      self.assertAllEqual(
          segments.peaks(values, invalidate_distance=99).eval(), [2])

  def test_peaks_array_filled_with_same_value(self):
    for value in (0, 42, 4.2):
      arr = tf.fill([100], value)
      with self.test_session():
        self.assertEmpty(segments.peaks(arr).eval())

  def test_peaks_one_segment(self):
    values = tf.constant([0, 0, 0, 0, 3, 0, 0, 0, 0])
    with self.test_session():
      self.assertAllEqual(segments.peaks(values).eval(), [4])


if __name__ == '__main__':
  tf.test.main()
