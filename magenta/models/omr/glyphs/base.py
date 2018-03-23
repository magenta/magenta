"""Base glyph classifier model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

# internal imports
import enum
import numpy as np

from magenta.models.omr.protobuf import musicscore_pb2


class GlyphsTensorColumns(enum.IntEnum):
  """The columns of the glyphs tensors.

  Glyphs should be held in a 2D tensor where the columns are the staff of the
  glyph, the vertical position on the staff, x coordinate, and glyph type.
  """

  STAFF_INDEX = 0
  Y_POSITION = 1
  X = 2
  TYPE = 3


class BaseGlyphClassifier(object):
  """The base glyph classifier model."""
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    """Base constructor for a glyph classifier.

    Attributes:
      staffline_extractor: Optional staffline extractor, if used for
        classification. If present, classification uses the scaled stafflines,
        and glyph x positions will be scaled back to page coordinates when
        constructing the Page. If None, no scaling is done.
    """
    self.staffline_extractor = None

  @abc.abstractmethod
  def get_detected_glyphs(self):
    """Detects glyphs in the image.

    Each glyph belongs to a staff, and has a y position numbered from 0 for the
    center staff line.

    Returns:
      A Tensor of glyphs, with shape (num_glyphs, 4). The columns are indexed by
        `GlyphsTensorColumns`. The glyphs will be sorted later, so they may be
        in any order.
    """
    pass

  def glyph_predictions_to_page(self, predictions):
    """Converts the glyph predictions to a Page message.

    Args:
      predictions: NumPy array which is equal to
        `self.get_detected_glyphs().eval()` (but multiple tensors are evaluated
        in a single run for efficiency.) Shape `(num_glyphs, 3)`.

    Returns:
      A `Page` message holding a single `StaffSystem`, with `Staff` messages
        that only hold `Glyph`s. Structural information is added to the page by
        `OMREngine`.
    """
    num_staves = (predictions[:, int(GlyphsTensorColumns.STAFF_INDEX)].max() + 1
                  if predictions.size else 0)

    def create_glyph(glyph):
      return musicscore_pb2.Glyph(
          x=glyph[GlyphsTensorColumns.X],
          y_position=glyph[GlyphsTensorColumns.Y_POSITION],
          type=glyph[GlyphsTensorColumns.TYPE])

    def generate_staff(staff_num):
      glyphs = predictions[
          predictions[:, int(GlyphsTensorColumns.STAFF_INDEX)] == staff_num]
      # For determinism, sort glyphs by x, breaking ties by position (low to
      # high).
      glyph_order = np.lexsort(
          glyphs[:, [GlyphsTensorColumns.Y_POSITION, GlyphsTensorColumns.X]].T)
      glyphs = glyphs[glyph_order]
      return musicscore_pb2.Staff(glyph=map(create_glyph, glyphs))

    return musicscore_pb2.Page(system=[
        musicscore_pb2.StaffSystem(staff=map(generate_staff, range(num_staves)))
    ])
