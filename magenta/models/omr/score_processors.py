"""Processors that need to visit each page of the score in one pass.

These are intended for detecting musical elements, where musical context may
span staff systems and pages (e.g. the time signature). Musical elements (e.g.
notes) are added to the `Score` message directly.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.omr.score import reader


def create_processors():
  yield reader.ScoreReader()


def process(score):
  """Processes a Score.

  Detects notes in the Score, and returns the Score in place.

  Args:
    score: A `Score` message.

  Returns:
    A `Score` message with `Note`s added to the `Glyph`s where applicable.
  """
  for processor in create_processors():
    score = processor(score)
  return score
