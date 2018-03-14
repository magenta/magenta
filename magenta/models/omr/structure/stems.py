"""Stem detection.

A stem detector takes stem candidates from `ColumnBasedVerticals`, which are
vertical lines with a height close to the expected height of a stem.

The distance is computed from each notehead to each stem, and the notehead is
assigned to the closest stem if the distance is below a threshold. The distance
is based on the coordinate where the notehead would ideally lie if it belongs
to the stem. First, the glyph y is clamped to the range of the stem, because the
center of the notehead should not be above or below the stem. Next, if the glyph
is left of the stem, the ideal x is a constant left of the stem, and similar if
it is right of the stem. This is because the left or right side of the notehead
should touch the stem, so they are ideally a fixed distance away.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np

from magenta.models.omr.glyphs import glyph_types
from magenta.models.omr.protobuf import musicscore_pb2

# The minimum height of a stem, as a multiple of the staffline distance.
_MIN_STEM_HEIGHT_STAFFLINE_DISTANCE = 2.5
# The expected horizontal distance from the notehead to the stem.
_STEM_NOTEHEAD_HORIZONTAL_STAFFLINE_DISTANCE = 0.5
# The maximum Euclidean distance from a notehead to its ideal position for a
# given stem (see module docstring).
_STEM_NOTEHEAD_DISTANCE_STAFFLINE_DISTANCE = 0.5


class Stems(object):
  """Stem detector."""

  def __init__(self, structure):
    """Constructs the stem detector.

    Args:
      structure: A computed structure.

    Raises:
      ValueError: If structure.is_computed() is false.
    """
    if not structure.is_computed():
      raise ValueError("Run Structure.compute() before passing it here")
    self.staff_detector = structure.staff_detector
    staffline_distance = np.mean(self.staff_detector.staffline_distance)
    verticals = structure.verticals
    self.stem_candidates = _get_stem_candidates(staffline_distance, verticals)

  def apply(self, page):
    """Detects stems on the page.

    Using `self.stem_candidates`, finds verticals that align with a notehead
    glyph, and adds the stems.

    Args:
      page: The Page message.

    Returns:
      The same page, updated with stems.
    """
    for system in page.system:
      for staff, staff_ys in zip(system.staff,
                                 self.staff_detector.staves_interpolated_y):
        allowed_distance = np.multiply(
            _STEM_NOTEHEAD_DISTANCE_STAFFLINE_DISTANCE,
            staff.staffline_distance)
        expected_horizontal_distance = np.multiply(
            _STEM_NOTEHEAD_HORIZONTAL_STAFFLINE_DISTANCE,
            staff.staffline_distance)
        for glyph in staff.glyph:
          if glyph_types.is_stemmed_notehead(glyph):
            glyph_y = (staff_ys[glyph.x] -
                       glyph.y_position * staff.staffline_distance / 2.0)
            # Compute the ideal coordinates for the glyph to be assigned to each
            # stem.

            # Clip the glyph_y to the stem start and end y to get the ideal y.
            ideal_y = np.clip(glyph_y, self.stem_candidates[:, 0, 1],
                              self.stem_candidates[:, 1, 1])
            # If the glyph is left of the stem, subtract the expected distance
            # from the stem x; otherwise, add it.
            ideal_x = self.stem_candidates[:, 0, 0] + np.where(
                glyph.x < self.stem_candidates[:, 0, 0],
                -expected_horizontal_distance, expected_horizontal_distance)
            stem_distance = np.linalg.norm(
                np.c_[ideal_x - glyph.x, ideal_y - glyph_y], axis=1)
            stem = np.argmin(stem_distance)
            if stem_distance[stem] <= allowed_distance:
              stem_coords = self.stem_candidates[stem]
              glyph.stem.CopyFrom(
                  musicscore_pb2.LineSegment(
                      start=musicscore_pb2.Point(
                          x=stem_coords[0, 0], y=stem_coords[0, 1]),
                      end=musicscore_pb2.Point(
                          x=stem_coords[1, 0], y=stem_coords[1, 1])))
    return page


def _get_stem_candidates(staffline_distance, verticals):
  heights = verticals.lines[:, 1, 1] - verticals.lines[:, 0, 1]
  return verticals.lines[np.greater_equal(
      heights, staffline_distance * _MIN_STEM_HEIGHT_STAFFLINE_DISTANCE)]
