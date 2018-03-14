"""Splits the single StaffSystem into multiple StaffSystems with bars."""
# TODO(ringwalt): Detect double barlines (with the expected distance between
# them) as one DOUBLE_BAR.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np
from six import moves

from magenta.models.omr.protobuf import musicscore_pb2


class Barlines(object):
  """Staff system and barline detector."""

  def __init__(self, structure, close_barline_threshold=None):
    barline_valid, self.barline_staff_start, self.barline_staff_end = (
        assign_barlines_to_staves(
            barline_x=structure.verticals.lines[:, :, 0].mean(
                axis=1).astype(int),
            barline_y0=structure.verticals.lines[:, 0, 1],
            barline_y1=structure.verticals.lines[:, 1, 1],
            staff_detector=structure.staff_detector))
    self.barlines = structure.verticals.lines[barline_valid]
    self.close_barline_threshold = (
        close_barline_threshold or
        np.median(structure.staff_detector.staffline_distance) * 4)

  def apply(self, page):
    """Splits the staves in the page into systems with barlines."""
    assert len(page.system) == 1
    systems_map = dict(
        (i, (i, i)) for i in moves.range(len(page.system[0].staff))
    )
    for start, end in zip(self.barline_staff_start, self.barline_staff_end):
      for staff in moves.range(start, end + 1):
        start = min(start, systems_map[staff][0])
        end = max(end, systems_map[staff][1])
      for staff in moves.range(start, end + 1):
        systems_map[staff] = (start, end)
    system_inds = sorted(set(systems_map.values()), key=lambda x: x[0])
    staves = page.system[0].staff
    systems = [
        musicscore_pb2.StaffSystem(staff=staves[start:end + 1])
        for (start, end) in system_inds
    ]
    self._assign_barlines(systems)
    return musicscore_pb2.Page(system=systems)

  def _assign_barlines(self, systems):
    """Assigns each barline to a system.

    Args:
      systems: The list of StaffSystem messages.
    """
    system_start = 0
    for system in systems:
      system_end = system_start + len(system.staff) - 1
      selected_barlines = set()
      blacklist_x = self._get_blacklist_x(system)
      for i in moves.range(len(self.barlines)):
        barline_x = self.barlines[i, 0, 0]
        start = self.barline_staff_start[i]
        end = self.barline_staff_end[i]
        if (not blacklist_x[barline_x] and
            system_start <= start <= end <= system_end):
          # Get the selected barlines which are close enough to the current
          # barline that they are probably a duplicate.
          close_barlines = [
              other_barline for other_barline in selected_barlines
              if abs(self.barlines[other_barline, 0, 0] - barline_x) <
              self.close_barline_threshold
          ]

          def get_span(barline):
            return (self.barline_staff_end[barline] -
                    self.barline_staff_start[barline])

          # Assumes all barlines span the entire staff system.
          # Don't add a barline if we've already seen a duplicate unless it
          # spans more staves than the currently selected one.
          # TODO(ringwalt): This works for piano scores, but not multi-part
          # scores, which have one barline spanning the entire staff system at
          # the beginning and then one barline per staff for the following
          # measures. Make this more robust.
          if (all(end - start >= get_span(other_barline)
                  for other_barline in selected_barlines) and
              all(end - start > get_span(other_barline)
                  for other_barline in close_barlines)):
            selected_barlines.difference_update(close_barlines)
            selected_barlines.add(i)
      barline_xs = sorted(self.barlines[barline, 0, 0]
                          for barline in selected_barlines)
      system.bar.extend(
          musicscore_pb2.StaffSystem.Bar(
              x=x, type=musicscore_pb2.StaffSystem.Bar.STANDARD_BAR)
          for x in barline_xs)

      system_start = system_end + 1

  def _get_blacklist_x(self, system):
    """Computes the x coordinates that are blacklisted for barlines.

    Barlines cannot be too close to a detected stem, because stems at a certain
    vertical position could be confused with barlines spanning a single staff.

    Args:
      system: The StaffSystem message.

    Returns:
      A boolean NumPy array. 1D and long enough to contain all of the barlines
      on the x axis. True for x coordinates where barlines are disallowed.
    """
    staffline_distance = np.median(
        [staff.staffline_distance for staff in system.staff]).astype(int)
    # Width needed to contain all of the barlines.
    barlines_width = (0 if self.barlines.size == 0 else
                      np.max(self.barlines[:, :, 0]) + 1)
    blacklist_x = np.zeros(barlines_width, np.bool)
    for staff in system.staff:
      for glyph in staff.glyph:
        if glyph.HasField('stem'):
          stem = glyph.stem
          blacklist_start = max(
              0,
              min(stem.start.x, stem.end.x) - staffline_distance)
          blacklist_end = min(
              barlines_width,
              max(stem.start.x, stem.end.x) + staffline_distance)
          blacklist_x[blacklist_start:blacklist_end] = True
    return blacklist_x


def assign_barlines_to_staves(barline_x, barline_y0, barline_y1,
                              staff_detector):
  """Chooses valid barlines for each staff.

  Args:
    barline_x: 1D array of length N. The barline x coordinates.
    barline_y0: 1D array of length N. The barline top y coordinates.
    barline_y1: 1D array of length N. The barline bottom y coordinates.
    staff_detector: A BaseStaffDetector, for reading the staffline distance.

  Returns:
    A tuple of:
    barline_valid: Boolean array of length N. Whether each of the input barlines
        was selected as a valid barline.
    barline_staff_start: Boolean array of length `K = barline_valid.sum()`. The
        staff index for the top of each valid barline.
    barline_staff_end: Boolean array of length K. The staff index for the bottom
        of each valid barline. Each entry is >= barline_staff_start.
  """
  # To be a barline, the start and end of the line have to be this close to
  # the start or end of the staff.
  max_distance_to_start_and_end_of_staff = (staff_detector.staffline_distance)
  # Compute the closest start and end staves for each vertical line.
  staff_starts = (
      staff_detector.staves_interpolated_y -
      2 * staff_detector.staffline_distance[:, None])
  barline_staff_start_distance = np.abs(
      barline_y0[None, :] - staff_starts[:, barline_x])
  barline_staff_start = np.argmin(barline_staff_start_distance, axis=0)

  # Barlines must be at most a single staffline distance away from the
  # expected start and end, which are the top line of the start staff and the
  # bottom line of the end staff.
  barline_valid = np.less_equal(
      np.min(barline_staff_start_distance, axis=0),
      max_distance_to_start_and_end_of_staff[barline_staff_start])

  # Check the closest end staff.
  staff_ends = (
      staff_detector.staves_interpolated_y +
      2 * staff_detector.staffline_distance[:, None])
  barline_staff_end_distance = np.abs(barline_y1 - staff_ends[:, barline_x])
  barline_staff_end = np.argmin(barline_staff_end_distance, axis=0)
  barline_valid &= np.less_equal(
      np.min(barline_staff_end_distance, axis=0),
      max_distance_to_start_and_end_of_staff[barline_staff_end])
  return (barline_valid, barline_staff_start[barline_valid],
          barline_staff_end[barline_valid])
