"""Tests for MusicXML evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os.path

# internal imports
from absl.testing import absltest
from lxml import etree
import pandas as pd
import pandas.util.testing as pd_testing
import six
from six import moves

from magenta.models.omr.evaluation import musicxml

from tensorflow.python.platform import resource_loader


class MusicXMLTest(absltest.TestCase):

  def testNotSimilar(self):
    self.assertEqual(musicxml._not_similar()['overall_score']['total'][0], 0)

  def testEmptyMeasures(self):
    a = etree.Element('score')
    a_part = etree.SubElement(a, 'part')
    measure = etree.SubElement(a_part, 'measure')
    attributes = etree.SubElement(measure, 'attributes')
    etree.SubElement(attributes, 'divisions').text = '24'
    etree.SubElement(a_part, 'measure')
    etree.SubElement(a_part, 'measure')
    b = etree.Element('score')
    b_part = etree.SubElement(b, 'part')
    measure = etree.SubElement(b_part, 'measure')
    attributes = etree.SubElement(measure, 'attributes')
    etree.SubElement(attributes, 'divisions').text = '24'
    etree.SubElement(b_part, 'measure')
    etree.SubElement(b_part, 'measure')
    similarity = musicxml.musicxml_similarity(a, b)
    self.assertEqual(len(similarity), 4)
    self.assertEqual(similarity['overall_score']['total'][0], 1.0)

  def testSomeMeasuresNotEqual(self):
    a = etree.Element('score')
    a_part = etree.SubElement(a, 'part')
    measure = etree.SubElement(a_part, 'measure')
    attributes = etree.SubElement(measure, 'attributes')
    etree.SubElement(attributes, 'divisions').text = '24'
    etree.SubElement(etree.SubElement(a_part, 'measure'), 'note')
    a_part = etree.SubElement(a, 'part')
    etree.SubElement(etree.SubElement(a_part, 'measure'), 'note')
    etree.SubElement(a_part, 'measure')

    b = etree.Element('score')
    b_part = etree.SubElement(b, 'part')
    measure = etree.SubElement(b_part, 'measure')
    attributes = etree.SubElement(measure, 'attributes')
    etree.SubElement(attributes, 'divisions').text = '24'
    etree.SubElement(measure, 'note')
    etree.SubElement(etree.SubElement(b_part, 'measure'), 'note')
    b_part = etree.SubElement(b, 'part')
    etree.SubElement(b_part, 'measure')
    etree.SubElement(etree.SubElement(b_part, 'measure'), 'note')

    pd_testing.assert_frame_equal(
        musicxml.musicxml_similarity(a, b),
        pd.DataFrame(
            [0.0, 1.0, 0.0, 0.0, 0.25],
            columns=[musicxml.OVERALL_SCORE],
            index=pd.MultiIndex.from_tuples(
                [(0, 0), (0, 1), (1, 0), (1, 1), ('total', '')],
                names=['staff', 'measure'])))

  def testIdentical(self):
    filename = six.text_type(
        os.path.join(resource_loader.get_data_files_path(),
                     '../testdata/IMSLP00747.golden.xml'))
    score = etree.fromstring(open(filename, 'rb').read())
    similarity = musicxml.musicxml_similarity(score, score)
    self.assertGreater(len(similarity), 1)
    self.assertEqual(similarity['overall_score']['total'][0], 1.0)

  def testMoreChangesLessSimilar(self):
    filename = six.text_type(
        os.path.join(resource_loader.get_data_files_path(),
                     '../testdata/TWO_MEASURE_SAMPLE.xml'))
    score = etree.fromstring(open(filename, 'rb').read())
    score2 = copy.deepcopy(score)
    durations2 = score2.findall('part/measure/note/duration')
    # Change the duration of one note by +1
    durations2[0].text = str(int(durations2[0].text) + 1)
    score3 = copy.deepcopy(score2)
    octaves3 = score3.findall('part/measure/note/pitch/octave')
    # Change the octave of another note by +1
    octaves3[1].text = str(int(octaves3[1].text) + 1)
    similarity11 = musicxml.musicxml_similarity(score, score)
    similarity12 = musicxml.musicxml_similarity(score, score2)
    similarity13 = musicxml.musicxml_similarity(score, score3)

    # Score 2 should be less similar than 1 to itself
    self.assertLess(similarity12['overall_score']['total'][0],
                    similarity11['overall_score']['total'][0])
    # Score 3 should be less similar than 2 to 1
    self.assertLess(similarity13['overall_score']['total'][0],
                    similarity12['overall_score']['total'][0])

  def testLevenshteinDistance(self):

    def levenshtein(a, b):
      return musicxml.levenshtein(a, b, musicxml.exact_match_distance)

    self.assertEqual(levenshtein('sky', 'sky'), 0)
    self.assertEqual(levenshtein('sky', 'soy'), 1)
    self.assertEqual(levenshtein('sky', 'skyll'), 2)
    self.assertEqual(levenshtein('sky', 's'), 2)
    self.assertEqual(levenshtein('sky', 'dig'), 3)
    self.assertEqual(levenshtein([(1, 2), (3, 4)], [(1, 2), (4, 3)]), 1)
    self.assertEqual(
        levenshtein([(1), (2), (3, (1, 2))], [(1, 2), (2), (3, (1, 3))]), 2)

  def testOptionalAlter(self):
    pitch = etree.Element('pitch')
    etree.SubElement(pitch, 'step').text = 'C'
    etree.SubElement(pitch, 'octave').text = '4'

    pitch_with_alter = copy.deepcopy(pitch)
    etree.SubElement(pitch_with_alter, 'alter').text = '0'
    self.assertEqual(
        musicxml.pitch_to_int(pitch), musicxml.pitch_to_int(pitch_with_alter))

    pitch_with_alter.find('alter').text = '-1'
    self.assertNotEqual(
        musicxml.pitch_to_int(pitch), musicxml.pitch_to_int(pitch_with_alter))

  def testPitchToInt(self):
    pitch_c4 = etree.Element('pitch')
    etree.SubElement(pitch_c4, 'step').text = 'C'
    etree.SubElement(pitch_c4, 'octave').text = '4'
    self.assertEqual(musicxml.pitch_to_int(pitch_c4), 48)

    pitch_bsharp3 = etree.Element('pitch')
    etree.SubElement(pitch_bsharp3, 'step').text = 'B'
    etree.SubElement(pitch_bsharp3, 'octave').text = '3'
    etree.SubElement(pitch_bsharp3, 'alter').text = '1'
    self.assertEqual(musicxml.pitch_to_int(pitch_bsharp3), 48)

    pitch_dflat4 = etree.Element('pitch')
    etree.SubElement(pitch_dflat4, 'step').text = 'D'
    etree.SubElement(pitch_dflat4, 'octave').text = '4'
    etree.SubElement(pitch_dflat4, 'alter').text = '-1'
    self.assertEqual(musicxml.pitch_to_int(pitch_dflat4), 49)

  def testNoteDistance_samePitch_sameDuration_isEqual(self):
    self.assertEqual(musicxml._note_distance((48, 3), (48, 3)), 0.0)

  def testNoteDistance_samePitch_differentDuration_fuzzyDuration(self):
    self.assertAlmostEqual(musicxml._note_distance((48, 3), (48, 4)), 0.125)
    self.assertAlmostEqual(musicxml._note_distance((48, 3), (48, 6)), 0.25)
    self.assertAlmostEqual(musicxml._note_distance((48, 3), (48, 8)), 0.3125)

  def testNoteDistance_differentPitch_sameDuration_fuzzyPitch(self):
    self.assertAlmostEqual(musicxml._note_distance((48, 3), (49, 3)), 0.125)
    self.assertAlmostEqual(musicxml._note_distance((48, 3), (46, 3)), 0.25)
    self.assertAlmostEqual(musicxml._note_distance((48, 3), (55, 3)), 0.5)

  def testNoteDistance_differentPitchAndDuration_fuzzyDistance(self):
    self.assertAlmostEqual(musicxml._note_distance((48, 3), (47, 4)), 0.25)
    self.assertAlmostEqual(musicxml._note_distance((48, 3), (50, 5)), 0.45)
    self.assertAlmostEqual(musicxml._note_distance((48, 3), (52, 6)), 0.75)
    self.assertAlmostEqual(musicxml._note_distance((48, 1), (53, 4)), 0.875)


class PartStavesTest(absltest.TestCase):

  def testDivisions_golden(self):
    """Test that <divisions> is propagated across all measures."""
    filename = six.text_type(
        os.path.join(resource_loader.get_data_files_path(),
                     '../testdata/IMSLP00747.golden.xml'))
    score = etree.fromstring(open(filename, 'rb').read())
    part_staves = musicxml.PartStaves(score)
    self.assertEqual(part_staves.num_partstaves(), 2)
    self.assertEqual(part_staves.num_measures(0), 22)
    self.assertEqual(part_staves.num_measures(1), 22)
    for i in moves.range(2):
      for j in moves.range(22):
        measure = part_staves.get_measure(i, j)
        self.assertEqual(measure.find('attributes/divisions').text, '8')

  def testAttributes_perStaff(self):
    score = etree.Element('score-partwise')
    part = etree.SubElement(score, 'part')
    measure = etree.SubElement(part, 'measure')
    attributes = etree.SubElement(measure, 'attributes')

    # Include the required <divisions>.
    etree.SubElement(attributes, 'divisions').text = '8'

    # Key signature for the first staff.
    key = etree.SubElement(attributes, 'key')
    key.attrib['number'] = '1'
    # Dummy value
    key.text = 'C major'

    # Key signature for the second staff.
    key = etree.SubElement(attributes, 'key')
    key.attrib['number'] = '2'
    # Dummy value
    key.text = 'G major'

    part_staves = musicxml.PartStaves(score)
    self.assertEqual(part_staves.num_partstaves(), 2)

    staff_1_measure = part_staves.get_measure(0, 0)
    self.assertEqual(len(staff_1_measure), 1)
    attributes = staff_1_measure.find('attributes')
    self.assertEqual(len(attributes), 2)
    self.assertEqual(etree.tostring(attributes[0]), b'<divisions>8</divisions>')
    self.assertEqual(etree.tostring(attributes[1]), b'<key>C major</key>')

    staff_2_measure = part_staves.get_measure(1, 0)
    self.assertEqual(len(staff_2_measure), 1)
    attributes = staff_2_measure.find('attributes')
    self.assertEqual(len(attributes), 2)
    self.assertEqual(etree.tostring(attributes[0]), b'<divisions>8</divisions>')
    self.assertEqual(etree.tostring(attributes[1]), b'<key>G major</key>')


if __name__ == '__main__':
  absltest.main()
