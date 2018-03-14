"""Fixes duplicate rests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
from magenta.models.omr.glyphs import glyph_types


class FixRepeatedRests(object):

  def apply(self, page):
    """Remove duplicate rests of the same type."""
    for system in page.system:
      for staff in system.staff:
        to_remove = []
        last_rest = None
        for glyph in staff.glyph:
          if (last_rest and glyph_types.is_rest(glyph) and
              last_rest.type == glyph.type and
              glyph.x - last_rest.x < staff.staffline_distance):
            to_remove.append(glyph)
          last_rest = glyph

        for glyph in to_remove:
          staff.glyph.remove(glyph)

    return page
