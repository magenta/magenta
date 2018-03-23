"""Adds staff location information to the Page.

The Page initially contains a single staff system with only glyphs, and this
adds the location of each staff.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# internal imports
from six import moves


class StaffProcessor(object):

  def __init__(self, structure, staffline_extractor):
    self.staff_detector = structure.staff_detector
    self.staffline_extractor = staffline_extractor

  def apply(self, page):
    """Adds staff location information to the Page message."""
    assert len(page.system) == 1, ('Page must initially have a single staff '
                                   'system')
    assert len(page.system[0].staff) == len(self.staff_detector.staves), (
        'Glyphs page must have the same number of staves as the staff detector')
    staves_arr = self.staff_detector.staves
    for i, staff in enumerate(page.system[0].staff):
      staff.staffline_distance = self.staff_detector.staffline_distance[i]
      for j in moves.range(staves_arr.shape[1]):
        if (0 < j and j + 1 < staves_arr.shape[1] and
            staves_arr[i, j - 1, 0] == staves_arr[i, j, 0] and
            staves_arr[i, j, 0] == staves_arr[i, j + 1, 0]):
          continue
        point = staff.center_line.add()
        point.x = staves_arr[i, j, 0]
        point.y = staves_arr[i, j, 1]

      # Scale the glyph x coordinates back for the original image.
      if self.staffline_extractor:
        # The height of an extracted slice of the image before scaling.
        staffline_orig_height = (
            staff.staffline_distance *
            self.staffline_extractor.staffline_distance_multiple)
        # The scale factor from the scaled staffline images to the original.
        staffline_scale = (
            staffline_orig_height / self.staffline_extractor.target_height)
        for glyph in staff.glyph:
          glyph.x = int(round(staffline_scale * glyph.x))
    return page
