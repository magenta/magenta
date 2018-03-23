"""Glyph y coordinate calculation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def glyph_y(staff, glyph):
  """Calculates the glyph y coordinate.

  Args:
    staff: A Staff, used for interpolating the staff center y coordinate.
    glyph: A Glyph on the Staff.

  Returns:
    The y coordinate of the glyph on the page.

  Raises:
    ValueError: If the glyph is not contained by the interval spanned by the
        staff on the x axis.
  """
  for point_a, point_b in zip(staff.center_line[:-1], staff.center_line[1:]):
    if point_a.x <= glyph.x < point_b.x:
      staff_center_y = point_a.y + ((point_b.y - point_a.y) *
                                    (glyph.x - point_a.x) //
                                    (point_b.x - point_a.x))
      # y positions count up (in the negative y direction).
      return staff_center_y - staff.staffline_distance * glyph.y_position // 2
  raise ValueError('Glyph (%s) is not contained by staff (%s)' %
                   (glyph, staff.center_line))
