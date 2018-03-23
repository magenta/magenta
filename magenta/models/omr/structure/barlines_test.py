"""Tests for stem detection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
from absl.testing import absltest
import numpy as np

from magenta.models.omr import structure
from magenta.models.omr.protobuf import musicscore_pb2
from magenta.models.omr.staves import base as staves_base
from magenta.models.omr.structure import barlines as barlines_module
from magenta.models.omr.structure import beams
from magenta.models.omr.structure import components
from magenta.models.omr.structure import verticals

Point = musicscore_pb2.Point  # pylint: disable=invalid-name


class BarlinesTest(absltest.TestCase):

  def testDummy(self):
    # Create a single staff, and a single vertical which is the correct height
    # of a stem. The vertical has x = 20 and goes from
    struct = structure.Structure(
        staff_detector=staves_base.ComputedStaves(
            staves=[[[10, 50], [90, 50]], [[11, 150], [91, 150]],
                    [[10, 250], [90, 250]], [[10, 350], [90, 350]]],
            staffline_distance=[12] * 4,
            staffline_thickness=2,
            staves_interpolated_y=[[50] * 100, [150] * 100, [250] * 100,
                                   [350] * 100]),
        beams=beams.ComputedBeams(np.zeros((0, 2, 2))),
        connected_components=components.ComputedComponents(np.zeros((0, 5))),
        verticals=verticals.ComputedVerticals(lines=[
            # Joins the first 2 staves.
            [[10, 50 - 12 * 2], [10, 150 + 12 * 2]],
            # Another barline, too close to the first one.
            [[12, 50 - 12 * 2], [12, 150 + 12 * 2]],
            # This barline is far enough, because the second barline was
            # skipped.
            [[13, 50 - 12 * 2], [13, 150 + 12 * 2]],
            # Single staff barlines are skipped.
            [[30, 50 - 12 * 2], [30, 50 + 12 * 2]],
            [[31, 150 - 12 * 2], [31, 150 + 12 * 2]],
            # Too close to a stem.
            [[70, 50 - 12 * 2], [70, 50 + 12 * 2]],
            # Too short.
            [[90, 50 - 12 * 2], [90, 50 + 12 * 2]],
            # Another barline which is kept.
            [[90, 50 - 12 * 2], [90, 150 + 12 * 2]],
            # Staff 1 has no barlines.
            # Staff 2 has 2 barlines.
            [[11, 350 - 12 * 2], [11, 350 + 12 * 2]],
            [[90, 350 - 12 * 2], [90, 350 + 12 * 2]],
        ]))
    barlines = barlines_module.Barlines(struct, close_barline_threshold=3)
    # Create a Page with Glyphs.
    input_page = musicscore_pb2.Page(system=[
        musicscore_pb2.StaffSystem(staff=[
            musicscore_pb2.Staff(
                staffline_distance=12,
                center_line=[
                    musicscore_pb2.Point(x=10, y=50),
                    musicscore_pb2.Point(x=90, y=50)
                ],
                glyph=[
                    # Stem is close to the last vertical on the first staff, so
                    # a barline will not be detected there.
                    musicscore_pb2.Glyph(
                        type=musicscore_pb2.Glyph.NOTEHEAD_FILLED,
                        x=60,
                        y_position=2,
                        stem=musicscore_pb2.LineSegment(
                            start=musicscore_pb2.Point(x=72, y=40),
                            end=musicscore_pb2.Point(x=72, y=80))),
                ]),
            musicscore_pb2.Staff(
                staffline_distance=12,
                center_line=[
                    musicscore_pb2.Point(x=10, y=150),
                    musicscore_pb2.Point(x=90, y=150)
                ]),
            musicscore_pb2.Staff(
                staffline_distance=12,
                center_line=[
                    musicscore_pb2.Point(x=10, y=250),
                    musicscore_pb2.Point(x=90, y=250)
                ]),
            musicscore_pb2.Staff(
                staffline_distance=12,
                center_line=[
                    musicscore_pb2.Point(x=10, y=350),
                    musicscore_pb2.Point(x=90, y=350)
                ]),
        ])
    ])
    page = barlines.apply(input_page)
    self.assertEqual(3, len(page.system))

    self.assertEqual(2, len(page.system[0].staff))
    self.assertItemsEqual([10, 13, 90], (bar.x for bar in page.system[0].bar))

    self.assertEqual(1, len(page.system[1].staff))
    self.assertEqual(0, len(page.system[1].bar))

    self.assertEqual(1, len(page.system[2].staff))
    self.assertEqual(2, len(page.system[2].bar))
    self.assertItemsEqual([11, 90], (bar.x for bar in page.system[2].bar))


if __name__ == "__main__":
  absltest.main()
