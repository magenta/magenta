"""Utility for glyph types.

Determines which modifiers may be attached to which glyphs (currently just
noteheads).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.omr.protobuf import musicscore_pb2


def is_notehead(glyph):
  return glyph.type in [
      musicscore_pb2.Glyph.NOTEHEAD_EMPTY, musicscore_pb2.Glyph.NOTEHEAD_FILLED,
      musicscore_pb2.Glyph.NOTEHEAD_WHOLE
  ]


def is_stemmed_notehead(glyph):
  return glyph.type in [
      musicscore_pb2.Glyph.NOTEHEAD_EMPTY, musicscore_pb2.Glyph.NOTEHEAD_FILLED
  ]


def is_beamed_notehead(glyph):
  return glyph.type == musicscore_pb2.Glyph.NOTEHEAD_FILLED


def is_dotted_notehead(glyph):
  # Any notehead can be dotted.
  return is_notehead(glyph)


def is_clef(glyph):
  return glyph.type in [
      musicscore_pb2.Glyph.CLEF_TREBLE, musicscore_pb2.Glyph.CLEF_BASS
  ]


def is_rest(glyph):
  return glyph.type in [
      musicscore_pb2.Glyph.REST_QUARTER,
      musicscore_pb2.Glyph.REST_EIGHTH,
      musicscore_pb2.Glyph.REST_SIXTEENTH,
  ]
