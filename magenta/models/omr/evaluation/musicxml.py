"""Simple MusicXML high-level diffing and evaluation.

This is a high-level evaluation for OMR because it compares musical elements,
instead of the individual glyphs and lines that are detected. It is also easier
to find MusicXML ground truth that corresponds exactly to a scanned score.

The evaluation result is a pandas DataFrame. This can eventually have multiple
columns with different evaluation metrics/subscores for different musical
elements for each measure (and the average for the whole score), which we can
choose between as needed.

This is considered high-level evaluation [1] because MusicXML does not have
information on every visual element (e.g. it has the stem direction and number
of beams, but not the start and end coordinates of the stem and beams). We also
want to evaluate emergent properties of OMR, such as note durations and voices,
instead of just counting the raw elements that are detected. This has the
drawback that one mistake in low-level symbol detection can have drastic effects
on the evaluation, but has the benefit that we can obtain thousands of MusicXML
scores. Low-level evaluation would require labeling the coordinates of all
elements on a given music score image, and there is not a public dataset for
printed music scores (MUSCIMA++ has handwritten music scores, which are
currently outside of our scope.)

[1] D. Byrd and J. G. Simonsen. Towards a standard testbed for optical music
    recognition: definitions, metrics, and page images, 2015.
    http://www.informatics.indiana.edu/donbyrd/OMRTestbed/
    OMRStandardTestbed1Mar2013.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import fractions

# internal imports
from lxml import etree
import pandas as pd
import six
from six import moves
from magenta.music import constants

OVERALL_SCORE = 'overall_score'


def musicxml_similarity(a, b):
  """Determines the similarity of two scores represented as musicXML strings.

  We currently assume the following:
  - The durations of each representation represent the same tempo.
    - i.e. the unit of measure tempo divisions is the same for both pieces.
  - Scores have an equal number of measures. They are dissimilar otherwise.
  - Corresponding parts have an equal number of staves.

  This currently accounts for:
  - Intra-measure note-to-note pitch and tempo distance, as an edit distance.
    - This can easily be generalized to weight differences based on distance,
      such as penalizing larger gaps in pitch or tempo.

  Args:
    a: a musicXML string
    b: a musicXML string

  Returns:
    A pd.DataFrame with scores on similarity between two scores. 1.0 means
    exactly the same, 0.0 means not similar at all.
    Example:
                        overall_score
      staff  measure
      0      0          0.750
             1          1.000
      total             0.875
  """
  if isinstance(a, six.binary_type):
    a = etree.fromstring(a)
  if isinstance(b, six.binary_type):
    b = etree.fromstring(b)

  a = PartStaves(a)
  b = PartStaves(b)

  # TODO(larryruili): Implement dissimilar measure count edit distance.
  if a.all_measure_counts() != b.all_measure_counts():
    return _not_similar()

  measure_similarities = []
  index = []
  for part_staff in moves.range(a.num_partstaves()):
    for measure_num in moves.range(a.num_measures(part_staff)):
      measure_similarities.append(
          measure_similarity(
              a.get_measure(part_staff, measure_num),
              b.get_measure(part_staff, measure_num)))
      index.append((part_staff, measure_num))
  df = pd.DataFrame(
      measure_similarities,
      columns=[OVERALL_SCORE],
      index=pd.MultiIndex.from_tuples(index, names=['staff', 'measure']))
  return df.append(
      pd.DataFrame(
          [df[OVERALL_SCORE].mean()],
          columns=[OVERALL_SCORE],
          index=pd.MultiIndex.from_tuples(
              [('total', '')], names=['staff', 'measure'])))


def _not_similar():
  """Returns a pd.DataFrame representing 0 similarity between scores."""
  return pd.DataFrame(
      [[0]],
      columns=[OVERALL_SCORE],
      index=pd.MultiIndex.from_product(
          [('total',), ('',)], names=['staff', 'measure']),)


def _index(num_staves, num_measures):
  return pd.MultiIndex.from_product(
      [range(num_staves), range(num_measures)], names=['staff', 'measure'])


class PartStaves(object):
  """Accessor for parts and staves in a MusicXML score.

  self.score: An etree rooted at <score>
  self.part_staves: A list of tuples (part number, staff number)
  """

  def __init__(self, score):
    self.score = score
    num_parts = len(score.findall('part'))
    part_staves = []
    for i in moves.range(num_parts):
      # XPath is 1-indexed.
      part = score.find('part[%d]' % (i + 1))

      def num_staves(part):
        yield 1
        for staves_tag in part.findall('measure/attributes/staves'):
          yield int(staves_tag.text)
        # Find all tags underneath any <attributes> tag. They may also include
        # the staff number as a "number" XML attribute.
        for attribute in part.findall('.//attributes//'):
          if 'number' in attribute.attrib:
            yield int(attribute.attrib['number'])

      num_staves = max(num_staves(part))
      for j in moves.range(num_staves):
        part_staves.append((i, j))
    self.part_staves = part_staves

  def get_measure(self, part_staff, measure_num):
    """Returns a single staff measure given a part and measure index.

    Args:
      part_staff: Index of the staff across all parts.
      measure_num: Index of the measure.

    Returns:
      A <measure> etree element.
    """
    part_num, staff_num = self.part_staves[part_staff]
    part = self.score.find('part[%d]' % (part_num + 1))
    measure = part.find('measure[%d]' % (measure_num + 1))
    staff_measure = etree.Element('measure')
    staff_measure.append(self._build_attributes(staff_num, measure))
    for elem in measure:
      if elem.tag != 'attributes':
        new_elem = _filter_for_staff(staff_num, elem)
        if new_elem is not None:
          staff_measure.append(new_elem)
    return staff_measure

  def _build_attributes(self, staff_num, orig_measure):
    """Builds a new <attributes> tag for the given measure.

    If the measure already contains an <attributes> tag, all children are copied
    as long as they do not have a <staff> tag that's incompatible with the given
    staff number.

    If the existing <attributes> don't contain a <divisions> tag, we take the
    most recent <divisions> from any measure on any part. The <divisions> tag is
    required at the start of the score.

    Args:
      staff_num: Index of the staff across all parts.
      orig_measure: The original <measure> tag, which may or may not contain its
        own <attributes>.

    Returns:
      A new <attributes> tag, keeping any applicable attributes from
      `orig_measure`, and the most recent <divisions>.
    """
    new_attributes = etree.Element('attributes')

    orig_attributes = orig_measure.find('attributes')
    if orig_attributes is not None:
      for attribute in orig_attributes:
        new_attribute = _filter_for_staff(staff_num, attribute)
        if new_attribute is not None:
          new_attributes.append(new_attribute)
    # Look for the most recent "divisions" tag.
    def find_divisions():
      divisions = None
      for part in orig_measure.getparent().getparent().findall('part'):
        for prev_measure in part.findall('measure'):
          prev_attrs = prev_measure.find('attributes')
          if (prev_attrs is not None and
              prev_attrs.find('divisions') is not None):
            divisions = prev_attrs.find('divisions')
          if prev_measure is orig_measure:
            # Done.
            if divisions is None:
              raise ValueError('<divisions> is required')
            # Copy the tag, or else when appending the tag to a new element, it
            # will be deleted from the old element.
            return copy.deepcopy(divisions)

    if new_attributes.find('divisions') is None:
      new_attributes.append(find_divisions())
    return new_attributes

  def num_partstaves(self):
    """Returns the total count of parts/staves."""
    return len(self.part_staves)

  def num_measures(self, part_staff):
    part_num, _ = self.part_staves[part_staff]
    return len(self._get_part(part_num).findall('measure'))

  def _get_part(self, part_num):
    return self.score.find('part[%d]' % (part_num + 1))

  def all_measure_counts(self):
    return [
        self.num_measures(part_staff)
        for part_staff in moves.range(self.num_partstaves())
    ]


def _filter_for_staff(staff_num, elem):
  """Determines whether the element belongs to the given staff.

  Elements normally have a <staff> child, with a 1-indexed staff number.
  However, elements in <attributes> may use a "number" XML attribute instead
  (also 1-indexed).

  Args:
    staff_num: The 0-indexed staff index.
    elem: The XML element (a descendant of <measure>).

  Returns:
    A copied element with staff information removed, or None if the element does
    not belong to the given staff.
  """
  staff = elem.find('staff')
  new_elem = copy.deepcopy(elem)
  if staff is not None:
    for subelem in new_elem:
      if subelem.tag == 'staff':
        if subelem.text == str(staff_num + 1):
          # Got the correct "staff" tag.
          return new_elem
        else:
          # Incorrect "staff" value.
          return None
  # The "number" XML attribute can refer to the staff within <attributes>.
  if 'number' in new_elem.attrib:
    if new_elem.attrib['number'] == str(staff_num + 1):
      del new_elem.attrib['number']
      return new_elem
    else:
      # Incorrect "number" attribute.
      return None
  # No staff information, element should be copied to all staves.
  return copy.deepcopy(elem)


def measure_similarity(a, b):
  """Similarity metric between <measure> tags.

  This currently converts <note> tags into tuples of the form
  ((pitch letter, octave number, alteration number), duration),
  then determines the edit distance between the two sequences of notes,
  normalized by the maximum edit distance possible.

  The similarity is simply 1 - (normalized edit distance).

  Args:
    a: A <measure> etree tag.
    b: A <measure> etree tag.

  Returns:
    A float between 0.0 (no similarity) and 1.0 (measures are equivalent).
  """
  a = measure_to_note_list(a)
  b = measure_to_note_list(b)
  scale = max(len(a), len(b))
  if scale == 0:
    return 1
  else:
    return 1 - (levenshtein(a, b) / scale)


def measure_to_note_list(measure):
  notes = measure.findall('note')
  note_list = []
  for note in notes:
    duration = duration_to_fraction(note.find('duration'), measure)
    pitch = pitch_to_int(note.find('pitch'))
    note_list.append((pitch, duration))
  return note_list


def pitch_to_int(pitch):
  """Converts a <pitch> tag to a an integer representation.

  Note is a character [A-G].
  Octave is an integer [0-9]
  Alter is an integer [(-2)-2]

  The formula is: pitch = 12 * octave + note_map[note] + alter
  Where note_map maps a note to its 12-tone representation, where C = 0, D = 2,
  and B = 12. We only represent white-key tones in this map because the alter is
  a separate dimension.

  This is a heuristic that treats A#4 the same as Bb4, which may not
  be the semantics we want for an optical recognition pipeline. However, it is
  a simple way of gauging 'how far' a note's pitch is from the other.

  Args:
    pitch: A <pitch> etree tag.

  Returns:
    An integer representation of the pitch, from 0 to 129, which is the number
    of semitones the pitch is above C0.
  """
  note_map = dict(zip('CDEFGAB', constants.MAJOR_SCALE))
  if pitch is None:
    return None
  note = pitch.find('step').text
  octave = int(pitch.find('octave').text)
  alter_tag = pitch.find('alter')
  alter = int(alter_tag.text) if alter_tag is not None else 0
  return 12 * octave + note_map[note] + alter


def duration_to_fraction(duration, measure):
  if duration is None:
    return None
  else:
    return fractions.Fraction(
        int(duration.text), int(measure.find('attributes/divisions').text))


def _note_distance(n1, n2):
  """Returns the distance between two notes.

  Pitch distance is a scaled absolute difference in pitch values between the two
  notes. If the difference is below 6 half-steps, the distance is proportional
  to difference / 8, else it's 1. This represents 'close-enough' semantics,
  where we reward pitches that are at least reasonably close to the target,
  since pitches can easily be very far off.

  Duration distance is simply the ratio between the absolute distance between
  representations, and the maximum duration between the notes. This represents
  'percent-correct' semantics, because all durations will be at least somewhat
  correct.

  The note distance is defined as the average between pitch distance and
  duration distance. Currently each distance has the same weight.

  Args:
    n1: a tuple of (pitch_int_representation, duration_int_representation)
    n2: a tuple of (pitch_int_representation, duration_int_representation)

  Returns:
    A float from 0 to 1 representing the distance between the two notes.
  """
  # If either pitch or duration is missing from either note, use exact equality.
  if not (n1[0] and n2[0]) or not (n1[1] and n2[1]):
    return n1 != n2

  max_pitch_diff = 4.
  pitch_diff = float(abs(n1[0] - n2[0]))
  if pitch_diff > max_pitch_diff:
    pitch_distance = 1
  else:
    pitch_distance = pitch_diff / max_pitch_diff

  duration1 = float(n1[1])
  duration2 = float(n2[1])
  duration_distance = abs(duration1 - duration2) / max(duration1, duration2, 1)
  return (pitch_distance + duration_distance) / 2


def levenshtein(s1, s2, distance=_note_distance):
  """Determines the edit distance between two sequences.

  Args:
    s1: A sequence.
    s2: A sequence.
    distance: A callable that accepts an element from each of s1 and s2, and
        returns a float distance metric between 0 and 1.

  Returns:
    The Levenshtein distance between the two sequences.
  """
  if len(s1) < len(s2):
    return levenshtein(s2, s1, distance)

  if not s2:
    return len(s1)

  previous_row = range(len(s2) + 1)
  for i, c1 in enumerate(s1):
    current_row = [i + 1]
    for j, c2 in enumerate(s2):
      insertions = previous_row[j + 1] + 1
      deletions = current_row[j] + 1
      substitutions = previous_row[j] + distance(c1, c2)
      current_row.append(min(insertions, deletions, substitutions))
    previous_row = current_row

  return previous_row[-1]


def exact_match_distance(n1, n2):
  """Metric for `levenshtein_distance` that requires an exact match."""
  return n1 != n2
