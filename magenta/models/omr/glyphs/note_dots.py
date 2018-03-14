"""Detects dots which are attached to noteheads.

Dots are round, solid, smaller than other glyphs, and are typically spaced so
that they don't intersect with staff lines. Therefore, we detect them from the
connected components. We determine that the component is round-ish and solid if
the area (black pixel count) is at least half the area of the bounds of the
component.

Candidate note dots are components that are round-ish and follow the expected
size. For each notehead, we look for candidate dots slightly to the right of the
note to assign to it.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np

from magenta.models.omr.glyphs import geometry
from magenta.models.omr.glyphs import glyph_types
from magenta.models.omr.protobuf import musicscore_pb2
from magenta.models.omr.structure import components

COMPONENTS = components.ConnectedComponentsColumns


class NoteDots(object):

  def __init__(self, structure):
    self.dots = _extract_dots(structure)

  def apply(self, page):
    """Detects note dots in the page.

    Dots must be to the right of a notehead.

    Args:
      page: A `Page` message.

    Returns:
      The same `Page`, with note dots added in place.
    """
    for system in page.system:
      for staff in system.staff:
        for glyph in staff.glyph:
          if glyph_types.is_dotted_notehead(glyph):
            x_min = glyph.x
            x_max = glyph.x + staff.staffline_distance * 3.
            y = geometry.glyph_y(staff, glyph)
            y_min = y - staff.staffline_distance / 2.
            y_max = y + staff.staffline_distance / 2.
            dots = self.dots[_is_in_range(x_min, self.dots[:, 0], x_max)
                             & _is_in_range(y_min, self.dots[:, 1], y_max)]
            glyph.dot.extend(
                musicscore_pb2.Point(x=dot[0], y=dot[1]) for dot in dots)

    return page


def _is_in_range(min_value, values, max_value):
  return np.logical_and(min_value <= values, values <= max_value)


def _extract_dots(structure):
  """Returns candidate note dots.

  Note dots must be connected components which are roundish (the area of the
  component's bounds is at least half full), and are the expected size.

  Args:
    structure: A computed `Structure`.

  Returns:
    A numpy array of shape `(N, 2)`. Each entry holds the center `(x, y)` of a
    candidate note dot.
  """
  min_height_width = structure.staff_detector.staffline_thickness + 1
  # TODO(ringwalt): Are note dots typically smaller in ossia parts?
  max_height_width = np.median(
      structure.staff_detector.staffline_distance) * 2 / 3
  connected_components = structure.connected_components.components
  width = connected_components[:, COMPONENTS.
                               X1] - connected_components[:, COMPONENTS.X0]
  height = connected_components[:, COMPONENTS.
                                Y1] - connected_components[:, COMPONENTS.Y0]
  is_full = np.greater_equal(connected_components[:, COMPONENTS.SIZE] * 2,
                             width * height)
  candidates = connected_components[
      is_full
      & _is_in_range(min_height_width, width, max_height_width)
      & _is_in_range(min_height_width, height, max_height_width)]
  # pyformat would make this completely unreadable
  # pyformat: disable
  candidate_centers = (
      np.c_[
          (candidates[:, COMPONENTS.X0] + candidates[:, COMPONENTS.X1]) / 2,
          (candidates[:, COMPONENTS.Y0] + candidates[:, COMPONENTS.Y1]) / 2]
      .astype(int))
  return candidate_centers
