"""Tests for the staff detectors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

# internal imports
import numpy as np
import tensorflow as tf

from magenta.models.omr import image as omr_image
from magenta.models.omr import staves
from magenta.models.omr.staves import staffline_distance
from magenta.models.omr.staves import testing


class StaffDetectorsTest(tf.test.TestCase):

  def setUp(self):
    # The normal _MIN_STAFFLINE_DISTANCE_SCORE is too large for the small images
    # used in unit tests.
    self.old_min_staffline_distance_score = (
        staffline_distance._MIN_STAFFLINE_DISTANCE_SCORE)
    staffline_distance._MIN_STAFFLINE_DISTANCE_SCORE = 10

  def tearDown(self):
    staffline_distance._MIN_STAFFLINE_DISTANCE_SCORE = (
        self.old_min_staffline_distance_score)

  def test_single_staff(self):
    blank_row = [255] * 50
    staff_row = [255] * 4 + [0] * 42 + [255] * 4
    # Create an image with 5 staff lines, with a slightly noisy staffline
    # thickness and distance.
    image = np.asarray([blank_row] * 25
                       + [staff_row] * 2
                       + [blank_row] * 8
                       + [staff_row] * 3
                       + [blank_row] * 8
                       + [staff_row] * 3
                       + [blank_row] * 9
                       + [staff_row] * 2
                       + [blank_row] * 8
                       + [staff_row] * 2
                       + [blank_row] * 25,
                       np.uint8)
    for detector in self.generate_staff_detectors(image):
      with self.test_session() as sess:
        staves_arr, staffline_distances, staffline_thickness = sess.run(
            (detector.staves, detector.staffline_distance,
             detector.staffline_thickness))
      expected_y = 25 + 2 + 8 + 3 + 8 + 1  # y coordinate of the center line
      self.assertEqual(staves_arr.shape[0], 1,
                       'Expected single staff from detector %s. Got: %d' %
                       (detector, staves_arr.shape[0]))
      self.assertAlmostEqual(
          np.mean(staves_arr[0, :, 1]),  # average y position
          expected_y,
          delta=2.0)
      self.assertAlmostEqual(staffline_distances[0], 11, delta=1.0)
      self.assertLessEqual(staffline_thickness, 3)

  def test_corpus_image(self):
    # Test only the default staff detector (because projection won't detect all
    # staves).
    filename = os.path.join(tf.resource_loader.get_data_files_path(),
                            '../testdata/IMSLP00747-000.png')
    image_t = omr_image.decode_music_score_png(tf.read_file(filename))
    detector = staves.StaffDetector(image_t)
    with self.test_session() as sess:
      staves_arr, staffline_distances = sess.run(
          [detector.staves, detector.staffline_distance])
    self.assertAllClose(
        np.mean(staves_arr[:, :, 1], axis=1),  # average y position
        [413, 603, 848, 1040, 1286, 1476, 1724, 1915, 2162, 2354, 2604, 2795],
        atol=5)
    self.assertAllEqual(staffline_distances, [16] * 12)

  def test_staves_interpolated_y(self):
    # Test staff center line interpolation.
    # The sequence of (x, y) points always starts at x = 0 and ends at
    # x = width - 1.
    staff = tf.constant(
        np.array([[[0, 10], [5, 5], [11, 0], [15, 10], [20, 20], [23, 49]]],
                 np.int32))

    with self.test_session():
      line_y = testing.FakeStaves(tf.zeros([50, 24]),
                                  staff).staves_interpolated_y[0].eval()
    self.assertEquals(
        list(line_y), [
            10, 9, 8, 7, 6, 5, 4, 3, 3, 2, 1, 0, 2, 5, 8, 10, 12, 14, 16, 18,
            20, 30, 39, 49
        ])

  def test_staves_interpolated_y_empty(self):
    with self.test_session():
      self.assertAllEqual(
          testing.FakeStaves(tf.zeros([50, 25]), tf.zeros([0, 2, 2], np.int32))
          .staves_interpolated_y.eval().shape, [0, 25])

  def test_staves_interpolated_y_staves_dont_extend_to_edge(self):
    staff = tf.constant(np.array([[[5, 10], [12, 8]]], np.int32))
    with self.test_session():
      # The y values should extend past the endpoints to the edge of the image,
      # and should be equal to the y value at the nearest endpoint.
      self.assertAllEqual(
          testing.FakeStaves(tf.zeros([50, 15]),
                             staff).staves_interpolated_y[0].eval(),
          [10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 8, 8, 8, 8])

  def generate_staff_detectors(self, image):
    yield staves.ProjectionStaffDetector(image)
    yield staves.FilteredHoughStaffDetector(image)


if __name__ == '__main__':
  tf.test.main()
