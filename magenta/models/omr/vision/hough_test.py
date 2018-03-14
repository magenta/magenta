"""Tests for the Hough transform."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np
import tensorflow as tf

from magenta.models.omr.vision import hough


class HoughTest(tf.test.TestCase):

  def testHorizontalLines(self):
    image = np.asarray(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]])  # pyformat: disable
    thetas = np.asarray([np.pi / 2, np.pi / 4, 0, -np.pi / 4])
    with self.test_session() as sess:
      hough_bins = sess.run(hough.hough_lines(image, thetas))
    self.assertAllEqual(
        hough_bins,
        # theta pi/2 gives the horizontal projection (sum each row).
        [[0, 5, 0, 0, 7, 0, 4, 0, 0, 0, 0],
         # theta pi/4 rotates the lines counter-clockwise from horizontal, and
         # higher rho values go down and right into the image.
         [0, 1, 3, 2, 5, 2, 2, 1, 0, 0, 0],
         # theta 0 gives the vertical projection (sum each column).
         [2, 3, 3, 3, 2, 2, 1, 0, 0, 0, 0],
         # theta -pi/4 rotates the lines counter-clockwise from vertical, and
         # higher rho values go up and right away from the image.
         [5, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]])  # pyformat: disable

  def testHoughPeaks_verticalLines(self):
    image = np.asarray(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 0, 1, 0]])  # pyformat: disable
    # Test the full range of angles.
    thetas = np.linspace(-np.pi, np.pi, 101)
    hough_bins = hough.hough_lines(image, thetas)
    peak_rho_t, peak_theta_t = hough.hough_peaks(hough_bins, thetas)
    with self.test_session() as sess:
      peak_rho, peak_theta = sess.run((peak_rho_t, peak_theta_t))
    # Vertical line
    self.assertEqual(peak_rho[0], 1)
    self.assertAlmostEqual(peak_theta[0], 0)
    # Rotated line
    self.assertEqual(peak_rho[1], 3)
    self.assertAlmostEqual(peak_theta[1], -np.pi / 8, places=1)

  def testHoughPeaks_minval(self):
    image = np.asarray(
        [[0, 0, 0, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0]])  # pyformat: disable
    thetas = np.linspace(0, np.pi / 2, 17)
    hough_bins = hough.hough_lines(image, thetas)
    peak_rho_t, peak_theta_t = hough.hough_peaks(hough_bins, thetas, minval=2)
    with self.test_session() as sess:
      peak_rho, peak_theta = sess.run((peak_rho_t, peak_theta_t))
      self.assertEqual(peak_rho.shape, (1,))
      self.assertEqual(peak_theta.shape, (1,))

  def testHoughPeaks_minvalTooLarge(self):
    image = np.asarray(
        [[0, 0, 0, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0]])  # pyformat: disable
    thetas = np.linspace(0, np.pi / 2, 17)
    hough_bins = hough.hough_lines(image, thetas)
    peak_rho_t, peak_theta_t = hough.hough_peaks(hough_bins, thetas, minval=3.1)
    with self.test_session() as sess:
      peak_rho, peak_theta = sess.run((peak_rho_t, peak_theta_t))
      self.assertEqual(peak_rho.shape, (0,))
      self.assertEqual(peak_theta.shape, (0,))


if __name__ == '__main__':
  tf.test.main()
