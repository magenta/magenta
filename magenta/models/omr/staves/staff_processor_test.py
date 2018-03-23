"""Tests for the staff page processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

# internal imports
from absl.testing import absltest
import numpy as np

from magenta.models.omr import engine
from magenta.models.omr import structure as structure_module
from magenta.models.omr.glyphs import testing as glyphs_testing
from magenta.models.omr.protobuf import musicscore_pb2
from magenta.models.omr.staves import staff_processor
from magenta.models.omr.staves import testing as staves_testing


class StaffProcessorTest(absltest.TestCase):

  def testGetPage_x_scale(self):
    # Random staffline images matching the dimensions of PREDICTIONS.
    dummy_stafflines = np.random.random((2, 3, 5, 6))
    classifier = glyphs_testing.DummyGlyphClassifier(glyphs_testing.PREDICTIONS)
    image = np.random.randint(0, 255, (30, 20), dtype=np.uint8)
    staves = staves_testing.FakeStaves(
        image_t=image,
        staves_t=np.asarray([[[0, 10], [19, 10]], [[0, 20], [19, 20]]],
                            np.int32),
        staffline_distance_t=np.asarray([5, 20], np.int32),
        staffline_thickness_t=np.asarray(1, np.int32))
    structure = structure_module.create_structure(image,
                                                  lambda unused_image: staves)

    class DummyStafflineExtractor(object):
      """A placeholder for StafflineExtractor.

      It only contains the constants necessary to scale the x coordinates.
      """
      staffline_distance_multiple = 2
      target_height = 10

    omr = engine.OMREngine(lambda _: classifier)
    page = omr.process_image(
        # Feed in a dummy image. It doesn't matter because FakeStaves has
        # hard-coded staff values.
        np.random.randint(0, 255, (100, 100)),
        process_structure=False)
    page = staff_processor.StaffProcessor(structure,
                                          DummyStafflineExtractor()).apply(page)
    self.assertEqual(len(page.system[0].staff), 2)
    # The first staff has a staffline distance of 5.
    # The extracted staffline slices have an original height of
    # staffline_distance * staffline_distance_multiple (10), which equals
    # target_height here, so there is no scaling.
    self.assertEqual(
        musicscore_pb2.Staff(glyph=page.system[0].staff[0].glyph),
        glyphs_testing.GLYPHS_PAGE.system[0].staff[0])
    # Glyphs in the second staff have a scaled x coordinate.
    self.assertEqual(
        len(page.system[0].staff[1].glyph),
        len(glyphs_testing.GLYPHS_PAGE.system[0].staff[1].glyph))
    for glyph in glyphs_testing.GLYPHS_PAGE.system[0].staff[1].glyph:
      expected_glyph = copy.deepcopy(glyph)
      # The second staff has a staffline distance of 20. The extracted staffline
      # slice would be 4 times the size of the scaled staffline, so x
      # coordinates are scaled by 4. Also, the glyphs may be in a different
      # order.
      expected_glyph.x *= 4
      self.assertIn(expected_glyph, page.system[0].staff[1].glyph)


if __name__ == '__main__':
  absltest.main()
