from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

# internal imports
import numpy as np
import tensorflow as tf

from magenta.models.omr import image as image_module
from magenta.models.omr import structure


class StructureTest(tf.test.TestCase):

  def testCompute(self):
    filename = os.path.join(tf.resource_loader.get_data_files_path(),
                            '../testdata/IMSLP00747-000.png')
    image = image_module.decode_music_score_png(tf.read_file(filename))
    struct = structure.create_structure(image)
    with self.test_session():
      struct = struct.compute()
    self.assertEqual(np.int32, struct.staff_detector.staves.dtype)
    # Expected number of staves for the corpus image.
    self.assertEqual((12, 2, 2), struct.staff_detector.staves.shape)

    self.assertEqual(np.int32, struct.verticals.lines.dtype)
    self.assertEqual(3, struct.verticals.lines.ndim)
    self.assertEqual((2, 2), struct.verticals.lines.shape[1:])


if __name__ == '__main__':
  tf.test.main()
