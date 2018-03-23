"""Tests for staff removal."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

# internal imports
import numpy as np
import tensorflow as tf

from magenta.models.omr import image as omr_image
from magenta.models.omr import staves
from magenta.models.omr.staves import removal
from magenta.models.omr.staves import staffline_distance


class RemovalTest(tf.test.TestCase):

  def test_corpus_image(self):
    filename = os.path.join(tf.resource_loader.get_data_files_path(),
                            '../testdata/IMSLP00747-000.png')
    image_t = omr_image.decode_music_score_png(tf.read_file(filename))
    remover = removal.StaffRemover(staves.StaffDetector(image_t))
    with self.test_session() as sess:
      removed, image = sess.run([remover.remove_staves, image_t])
      self.assertFalse(np.allclose(removed, image))
      # If staff removal runs successfully, we should be unable to estimate the
      # staffline distance from the staves-removed image.
      est_staffline_distance, est_staffline_thickness = sess.run(
          staffline_distance.estimate_staffline_distance_and_thickness(removed))
      print(est_staffline_distance)
      self.assertAllEqual([], est_staffline_distance)
      self.assertEqual(-1, est_staffline_thickness)


if __name__ == '__main__':
  tf.test.main()
