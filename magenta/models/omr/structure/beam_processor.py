"""Adds Beams to notes with intersecting Stems.

First, detects beams that have enough area (black pixel count) proportionate to
their width to count as multiple beams. The beam coordinates are just repeated
in this case for each detected beam, because we don't know specifically where
each individual beam is.

Next, for each stem already attached to a note, we assign any intersecting beam
candidates to the note.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np

from magenta.models.omr.glyphs import glyph_types
from magenta.models.omr.protobuf import musicscore_pb2
from magenta.models.omr.structure import components

COLUMNS = components.ConnectedComponentsColumns


class BeamProcessor(object):

  def __init__(self, structure):
    self.beams = _maybe_duplicate_beams(
        structure.beams.beams,
        np.median(structure.staff_detector.staffline_distance))

  def apply(self, page):
    """Adds beams that intersect with note stems to the page.

    Beams should intersect with two or more stems. Beams are currently
    implemented as a bounding box, so we just see whether that box intersects
    with each stem.

    Args:
      page: A Page message.

    Returns:
      The same page, with `beam`s added to the `Glyph`s.
    """
    for system in page.system:
      for staff in system.staff:
        # Extend the beams by the staffline distance on either side. Beams may
        # end immediately at a stem, so give an extra allowance for that stem.
        extended_beams = self.beams.copy()
        extended_beams[:, COLUMNS.X0] -= staff.staffline_distance
        extended_beams[:, COLUMNS.X1] += staff.staffline_distance
        for glyph in staff.glyph:
          if glyph_types.is_beamed_notehead(glyph) and glyph.HasField('stem'):
            xs = [glyph.stem.start.x, glyph.stem.end.x]
            ys = [glyph.stem.start.y, glyph.stem.end.y]
            stem_bounding_box = np.asarray([[min(*xs), min(*ys)],
                                            [max(*xs), max(*ys)]])
            overlapping_beams = _get_overlapping_beams(stem_bounding_box,
                                                       extended_beams)
            glyph.beam.extend(
                musicscore_pb2.LineSegment(
                    start=musicscore_pb2.Point(
                        x=beam[COLUMNS.X0], y=beam[COLUMNS.Y0]),
                    end=musicscore_pb2.Point(
                        x=beam[COLUMNS.X1], y=beam[COLUMNS.Y1]))
                for beam in overlapping_beams)
    return page


def _get_overlapping_beams(stem, beams):
  """Filters beams that overlap with the stem.

  Args:
    stem: NumPy array `((x0, y0), (x1, y1))` representing the stem line.
    beams: NumPy array of shape `(num_beams, 2, 2)`. The line segment for every
        candidate beam.

  Returns:
    Filtered beams of shape `(num_filtered_beams, 2, 2)`. All of the beams which
        intersect with the given stem.
  """
  # The horizontal and vertical intervals of the stem line must match the
  # intervals that the beam covers. Broadcast the single stem against all of
  # the beams.
  x_overlaps = _do_intervals_overlap(stem[None, :, 0],
                                     beams[:, [COLUMNS.X0, COLUMNS.X1]])
  y_overlaps = _do_intervals_overlap(stem[None, :, 1],
                                     beams[:, [COLUMNS.Y0, COLUMNS.Y1]])
  return beams[np.logical_and(x_overlaps, y_overlaps)]


def _maybe_duplicate_beams(beams, staffline_distance):
  """Determines whether each candidate beam actually contains multiple beams.

  Beams are normally separated by a narrow space, but sometimes they can blur
  together. Example: https://imgur.com/2ompQAz.png

  The number of black pixels in a single beam is proportional to its width. If
  the total area of the component is a multiple of the expected area, repeat the
  beam to count as multiple beams.

  Args:
    beams: The connected component array with shape (N, 5). The values in the
        columns are determined in `components.ConnectedComponentsColumns`.
    staffline_distance: The scalar staffline distance (median from all staves).

  Returns:
    The beams array, possibly with some beam candidates repeated along the 0th
    axis.
  """

  def _estimate_num_beams(beam):
    width = beam[COLUMNS.X1] - beam[COLUMNS.X0]
    # Beams appear to typically be slightly shorter than the staffline distance.
    estimated_area_per_beam = width * staffline_distance * 0.75
    return max(1, np.round(beam[COLUMNS.SIZE] / estimated_area_per_beam))

  estimated_num_beams = list(map(_estimate_num_beams, beams))
  return np.repeat(beams, estimated_num_beams, axis=0)


def _do_intervals_overlap(intervals_a, intervals_b):
  """Whether the intervals overlap, pairwise.

  intervals_a and intervals_b should both have the same shape
  `(num_intervals, 2)`. For each interval from each argument, returns a boolean
  of whether the numeric intervals overlap.

  Args:
    intervals_a: Numeric NumPy array of shape `(num_intervals, 2)`.
    intervals_b: Numeric NumPy array of shape `(num_intervals, 2)`.

  Returns:
    Boolean NumPy array of length `num_intervals`.
  """

  def contained(points, intervals):
    return np.logical_and(
        np.less_equal(intervals[:, 0], points),
        np.less_equal(points, intervals[:, 1]))

  return np.logical_or(
      np.logical_or(
          contained(intervals_a[:, 0], intervals_b),
          contained(intervals_a[:, 1], intervals_b)),
      np.logical_or(
          contained(intervals_b[:, 0], intervals_a),
          contained(intervals_b[:, 1], intervals_a)))
