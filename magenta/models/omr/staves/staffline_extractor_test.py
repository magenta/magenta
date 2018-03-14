"""Tests for StafflineExtractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np
import tensorflow as tf

from magenta.models.omr import staves
from magenta.models.omr.staves import staffline_extractor


class StafflineExtractorTest(tf.test.TestCase):

  def testExtractStaff(self):
    # Small image with a single staff.
    image = np.asarray([[1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1],
                        [1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1]],
                       np.uint8) * 255
    image_t = tf.constant(image, name='image')
    detector = staves.ProjectionStaffDetector(image_t)
    # The staffline distance is 3, so use a target height of 6 to avoid scaling
    # the image.
    extractor = staffline_extractor.StafflineExtractor(
        image_t, detector, target_height=6, num_sections=9,
        staffline_distance_multiple=2)
    with self.test_session():
      stafflines = extractor.extract_staves().eval()
    assert stafflines.shape == (1, 9, 6, 7)
    # The top staff line is at a y-value of 2 because of rounding.
    assert np.array_equal(stafflines[0, 0],
                          np.concatenate((np.zeros((2, 7)), image[:4] / 255.0)))
    # The staff space is centered in the window.
    assert np.array_equal(stafflines[0, 3], image[3:9] / 255.0)


if __name__ == '__main__':
  tf.test.main()
