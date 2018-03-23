"""Detects section barlines, which are much thicker than normal barlines.

Section barlines appear as connected components which span the height of the
system, and are not too thick. They may have 2 repeat dots on one or both sides
of each staff (at y positions -1 and 1), which affect the barline type.
"""
# TODO(ringwalt): Get repeat dots from the components and adjust the barline
# type accordingly. Currently, assume all thick barlines are END_BAR.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np

from magenta.models.omr.protobuf import musicscore_pb2
from magenta.models.omr.structure import barlines
from magenta.models.omr.structure import components as components_module

Bar = musicscore_pb2.StaffSystem.Bar  # pylint: disable=invalid-name
COLUMNS = components_module.ConnectedComponentsColumns


class SectionBarlines(object):
  """Reads the connected components, and adds thick barlines to the page."""

  def __init__(self, structure):
    self.components = structure.connected_components.components
    self.staff_detector = structure.staff_detector

  def apply(self, page):
    """Detects thick section barlines from the connected components.

    These should be tall components that start and end near the start and end
    of two (possibly different) staves. We use the standard barlines logic to
    assign components to the nearest start and end staff. We filter for
    candidate barlines, whose start and end are sufficiently close to the
    expected values. We then filter again by whether the component width is
    within the expected values for section barlines.

    For each staff system, we take the section barlines that match exactly that
    system's staves. Any standard barlines that are too close to a new section
    barline are removed, and we merge the existing standard barlines with the
    new section barlines.

    Args:
      page: A Page message.

    Returns:
      The same Page message, with new section barlines added.
    """
    component_center_x = np.mean(
        self.components[:, [COLUMNS.X0, COLUMNS.X1]], axis=1).astype(int)
    # Take section barline candidates, whose start and end y values are close
    # enough to the staff start and end ys.
    component_is_candidate, candidate_start_staff, candidate_end_staff = (
        barlines.assign_barlines_to_staves(
            barline_x=component_center_x,
            barline_y0=self.components[:, COLUMNS.Y0],
            barline_y1=self.components[:, COLUMNS.Y1],
            staff_detector=self.staff_detector))
    candidates = self.components[component_is_candidate]
    candidate_center_x = component_center_x[component_is_candidate]
    del component_center_x

    # Filter again by the expected section barline width.
    component_width = candidates[:, COLUMNS.X1] - candidates[:, COLUMNS.X0]
    component_width_ok = np.logical_and(
        self._section_min_width() <= component_width,
        component_width <= self._section_max_width(candidate_start_staff))
    candidates = candidates[component_width_ok]
    candidate_center_x = candidate_center_x[component_width_ok]
    candidate_start_staff = candidate_start_staff[component_width_ok]
    candidate_end_staff = candidate_end_staff[component_width_ok]

    # For each existing staff system, consider only the candidates that match
    # exactly the system's start and end staves.
    start_staff = 0
    for system in page.system:
      staffline_distance = np.median(
          [staff.staffline_distance for staff in system.staff]).astype(int)
      candidate_covers_staff_system = np.logical_and(
          candidate_start_staff == start_staff,
          candidate_end_staff + 1 == start_staff + len(system.staff))
      # Calculate the x coordinates of all section barlines to keep.
      section_bar_x = candidate_center_x[candidate_covers_staff_system]
      # Extract the existing bar x coordinates and types for merging.
      existing_bar_type = {bar.x: bar.type for bar in system.bar}
      existing_bars = np.asarray([bar.x for bar in system.bar])
      # Merge the existing barlines and section barlines.
      if existing_bars.size and section_bar_x.size:
        # Filter the existing bars by whether they are far enough from a new
        # section barline. Section barlines override the existing standard
        # barlines.
        existing_bars_ok = np.greater(
            np.min(
                np.abs(existing_bars[:, None] - section_bar_x[None, :]),
                axis=1), staffline_distance * 4)
        existing_bars = existing_bars[existing_bars_ok]

      # Merge the existing barlines which we kept, and the new section barlines
      # (which are assumed to be type END_BAR), in sorted order.
      bars = sorted(
          [Bar(x=x, type=existing_bar_type[x]) for x in existing_bars] +
          [Bar(x=x, type=Bar.END_BAR) for x in section_bar_x],
          key=lambda bar: bar.x)
      # Update the staff system.
      system.ClearField('bar')
      system.bar.extend(bars)

      start_staff += len(system.staff)
    return page

  def _section_min_width(self):
    return self.staff_detector.staffline_thickness * 3

  def _section_max_width(self, staff_index):
    return self.staff_detector.staffline_distance[staff_index] * 2


class MergeStandardAndBeginRepeatBars(object):
  """Detects a begin repeat at the beginning of the staff system.

  Typically, a begin repeat bar on a new line will be preceded by a standard
  barline, clef, and key signature. We can override a standard bar with a
  section bar if they are close together, but this distance is typically closer
  than the two bars are in this case.

  We want the two bars to be replaced by a single begin repeat bar where we
  actually found the first bar, because we want the clef, key signature, and
  notes to be a single measure.

  Because we don't yet detect repeat dots, and all non-STANDARD barlines are
  detected as END_BAR, we accept any non-STANDARD barlines for the second bar.
  """

  def __init__(self, structure):
    self.staff_detector = structure.staff_detector

  def apply(self, page):
    for system in page.system:
      if (len(system.bar) > 1 and system.bar[0].type == Bar.STANDARD_BAR and
          system.bar[1].type != Bar.STANDARD_BAR):
        staffline_distance = np.median(
            [staff.staffline_distance for staff in system.staff])
        if system.bar[1].x - system.bar[0].x < staffline_distance * 12:
          system.bar[0].type = system.bar[1].type
          del system.bar[1]
    return page
