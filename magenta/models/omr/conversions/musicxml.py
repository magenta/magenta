"""Score to MusicXML conversion."""
# TODO(ringwalt): Key signature
# TODO(ringwalt): Chords
# TODO(ringwalt): Stems--MusicXML supports "up" or "down".
# TODO(ringwalt): Accurate layout of pages, staves, and measures.
# TODO(ringwalt): Barline types.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import re

# internal imports
import librosa
from lxml import etree
from six import moves

from magenta.models.omr.protobuf import musicscore_pb2
from magenta.models.omr.score import measures

DOCTYPE = ('<!DOCTYPE score-partwise PUBLIC\n'
           '    "-//Recordare//DTD MusicXML 3.0 Partwise//EN"\n'
           '    "http://www.musicxml.org/dtds/partwise.dtd">\n')
MUSICXML_VERSION = '3.0'

# Number of divisions (duration units) per quarter note.
DIVISIONS = 1024

# TODO(ringwalt): Detect the actual time signature.
TIME_SIGNATURE = etree.Element('time', symbol='common')
etree.SubElement(TIME_SIGNATURE, 'beats').text = '4'
etree.SubElement(TIME_SIGNATURE, 'beat-type').text = '4'

# Note types.
HALF = 'half'
WHOLE = 'whole'
# Indexed by the number of beams on a filled note. Each beam halves the duration
# of the note.
FILLED = [
    'quarter', 'eighth', '16th', '32nd', '64th', '128th', '256th', '512th',
    '1024th'
]

# Maps the ASCII accidental names to the actual pitch alteration.
ACCIDENTAL_TO_ALTER = {'': 0, '#': 1, 'b': -1}


def score_to_musicxml(score):
  """Converts a `tensorflow.magenta.omr.Score` to MusicXML.

  Args:
    score: The OMR score.

  Returns:
    XML text.
  """
  musicxml = MusicXMLScore(score)
  measure_num = 0
  for page in score.page:
    for system in page.system:
      system_measures = measures.Measures(system)
      for system_measure_num in moves.range(system_measures.size()):
        for staff_num, staff in enumerate(system.staff):
          # Produce the measure, even if there are no glyphs.
          measure = musicxml.get_measure(staff_num, measure_num)

          for glyph in staff.glyph:
            if system_measures.get_measure(glyph) == system_measure_num:
              clef = _glyph_to_clef(glyph)
              if clef is not None:
                attributes = _get_attributes(measure)
                if attributes.find('clef') is not None:
                  attributes.remove(attributes.find('clef'))
                attributes.append(clef)
              note = _glyph_to_note(glyph)
              if note is not None:
                measure.append(note)
        measure_num += 1
  # Add <divisions> and <time> to each part.
  for part in musicxml.score:
    # XPath indexing is 1-based.
    measure = part.find('measure[1]')
    if measure is not None:
      attributes = _get_attributes(measure, position=0)
      etree.SubElement(attributes, 'divisions').text = str(DIVISIONS)
      attributes.append(copy.deepcopy(TIME_SIGNATURE))
  return musicxml.to_string()


_TREBLE_CLEF = etree.Element('clef')
etree.SubElement(_TREBLE_CLEF, 'sign').text = 'G'
etree.SubElement(_TREBLE_CLEF, 'line').text = '2'
_BASS_CLEF = etree.Element('clef')
etree.SubElement(_BASS_CLEF, 'sign').text = 'F'
etree.SubElement(_BASS_CLEF, 'line').text = '4'


def _get_attributes(measure, position=-1):
  """Gets or creates an `<attributes>` tag in the `<measure>` tag.

  If the child of `measure` at the given `position` is not an `<attributes>`
  tag, creates a new `<attributes>` tag, appends, and returns it.
  `position == -1` will get or insert attributes at the end of the measure.

  Args:
    measure: A `<measure>` etree tag.
    position: The index where the attributes should be found or inserted.

  Returns:
    An `<attributes>` etree tag.
  """
  if len(measure) and measure[position].tag == 'attributes':
    return measure[position]
  else:
    attributes = etree.Element('attributes')
    measure.insert(position, attributes)
    return attributes


def _glyph_to_clef(glyph):
  """Converts a `Glyph` to a `<clef>` tag.

  Args:
    glyph: A `tensorflow.magenta.omr.Glyph` message.

  Returns:
    An etree `<clef>` tag, or `None` if the glyph is not a clef.
  """
  if glyph.type == musicscore_pb2.Glyph.CLEF_TREBLE:
    return copy.deepcopy(_TREBLE_CLEF)
  elif glyph.type == musicscore_pb2.Glyph.CLEF_BASS:
    return copy.deepcopy(_BASS_CLEF)
  else:
    return None


def _glyph_to_note(glyph):
  """Converts a `Glyph` message to a `<note>` tag.

  Args:
    glyph: A `tensorflow.magenta.omr.Glyph` message. The glyph type should be
        one of `NOTEHEAD_*`.

  Returns:
    An etree `<note>` tag, or `None` if the glyph is not a notehead.

  Raises:
    ValueError: If the note duration is not a multiple of `1 / DIVISIONS`.
  """
  if not glyph.HasField('note'):
    return None
  note = etree.Element('note')
  if glyph.type == musicscore_pb2.Glyph.NOTEHEAD_EMPTY:
    note_type = HALF
  elif glyph.type == musicscore_pb2.Glyph.NOTEHEAD_WHOLE:
    note_type = WHOLE
  else:
    index = min(len(FILLED), len(glyph.beam))
    note_type = FILLED[index]
  etree.SubElement(note, 'type').text = note_type
  duration = DIVISIONS * (glyph.note.end_time - glyph.note.start_time)
  if not duration.is_integer():
    raise ValueError('Duration is not an integer: ' + str(duration))
  etree.SubElement(note, 'duration').text = str(int(duration))
  pitch_match = re.match('([A-G])([#b]?)([0-9]+)',
                         librosa.midi_to_note(glyph.note.pitch))
  pitch = etree.SubElement(note, 'pitch')
  etree.SubElement(pitch, 'step').text = pitch_match.group(1)
  etree.SubElement(
      pitch, 'alter').text = str(ACCIDENTAL_TO_ALTER[pitch_match.group(2)])
  etree.SubElement(pitch, 'octave').text = pitch_match.group(3)
  return note


class MusicXMLScore(object):
  """Manages the parts and measures of the MusicXML score.

  Provides access to parts and measures by index, creating new parts and
  measures as needed.
  """

  def __init__(self, omr_score):
    num_parts = max(len(system.staff)
                    for page in omr_score.page
                    for system in page.system)
    part_list = _create_part_list(num_parts)
    self.score = etree.Element('score-partwise', version=MUSICXML_VERSION)
    self.score.append(part_list)

  def get_measure(self, part_ind, measure_ind):
    while len(self.score.findall('part')) <= part_ind:
      next_part_ind = len(self.score.findall('part'))
      etree.SubElement(self.score, 'part', id=_get_part_id(next_part_ind))
    # XPath indexing is 1-based.
    part = self.score.find('part[%d]' % (part_ind + 1))
    while len(part.findall('measure')) <= measure_ind:
      next_measure_ind = len(part.findall('measure'))
      etree.SubElement(part, 'measure', number=str(next_measure_ind + 1))
    return part.find('measure[%d]' % (measure_ind + 1))

  def to_string(self):
    return etree.tostring(
        self.score.getroottree(),
        pretty_print=True,
        xml_declaration=True,
        encoding='UTF-8',
        doctype=DOCTYPE)


def _create_part_list(num_parts):
  part_list = etree.Element('part-list')
  for part_num in moves.range(1, num_parts + 1):
    score_part = etree.SubElement(part_list, 'score-part')
    score_part.set('id', 'P%d' % part_num)
    etree.SubElement(score_part, 'part-name').text = 'Part %d' % part_num
  return part_list


def _get_part_id(part_ind):
  return 'P%d' % (part_ind + 1)
