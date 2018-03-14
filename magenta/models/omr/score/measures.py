"""Represents the measures of a staff system.

Converts bar x coordinates to a series of measures, with the x interval covered
by each measure.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.omr.protobuf import musicscore_pb2

# sys.maxint overflows the int32 proto field.
MEASURE_MAX_X = 2**31 - 1


class Measures(object):
  """Represents the measures of a staff system."""

  def __init__(self, staff_system):
    self.bars = list(_get_bar_intervals(staff_system))

  def size(self):
    """Returns the number of measures in the staff system.

    Returns:
      The number of measures.
    """
    return len(self.bars)

  def get_measure(self, glyph):
    """Gets the measure number of a `tensorflow.magenta.omr.Glyph`.

    Args:
      glyph: A `Glyph` message.

    Returns:
      The measure index, or -1 if it lies outside of the measures.
    """
    for i, (start_bar, end_bar) in enumerate(self.bars):
      if start_bar.x <= glyph.x < end_bar.x:
        return i
    return -1


def _get_bar_intervals(staff_system):
  if not staff_system.bar:
    # TODO(ringwalt): Store the image dimensions in the Page message, so that we
    # can use the actual width as the end of the measure.
    yield (musicscore_pb2.StaffSystem.Bar(x=0),
           musicscore_pb2.StaffSystem.Bar(x=MEASURE_MAX_X))
  elif len(staff_system.bar) == 1:
    # Single barline is at the beginning of the staff.
    yield staff_system.bar[0], musicscore_pb2.StaffSystem.Bar(x=MEASURE_MAX_X)
  else:
    for start, end in zip(staff_system.bar[:-1], staff_system.bar[1:]):
      yield start, end
