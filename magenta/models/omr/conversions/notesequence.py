"""Converts an OMR `Score` to a `NoteSequence`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.omr.protobuf import musicscore_pb2
from magenta.protobuf import music_pb2


def score_to_notesequence(score):
  """Score to NoteSequence conversion.

  Args:
    score: A `tensorflow.magenta.omr.Score` message.

  Returns:
    A `tensorflow.magenta.NoteSequence` message containing the notes in the
    score.
  """
  return music_pb2.NoteSequence(notes=list(_score_notes(score)))


def page_to_notesequence(page):
  return score_to_notesequence(musicscore_pb2.Score(page=[page]))


def _score_notes(score):
  for page in score.page:
    for system in page.system:
      for staff in system.staff:
        for glyph in staff.glyph:
          if glyph.HasField('note'):
            yield glyph.note
