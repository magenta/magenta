"""Simple end-to-end test for OMR on the sample image."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

# internal imports
from absl.testing import absltest
import librosa
from lxml import etree
import numpy as np
from PIL import Image

from magenta.models.omr import conversions
from magenta.models.omr import engine
from magenta.protobuf import music_pb2

from tensorflow.python.platform import resource_loader


class OmrEndToEndTest(absltest.TestCase):

  def setUp(self):
    self.engine = engine.OMREngine()

  def testNoteSequence(self):
    filename = os.path.join(resource_loader.get_data_files_path(),
                            'testdata/IMSLP00747-000.png')
    notes = self.engine.run(filename, output_notesequence=True)
    # TODO(ringwalt): Fix the extra note that is detected before the actual
    # first eighth note.
    self.assertEqual(librosa.note_to_midi('C4'), notes.notes[1].pitch)
    self.assertEqual(librosa.note_to_midi('D4'), notes.notes[2].pitch)
    self.assertEqual(librosa.note_to_midi('E4'), notes.notes[3].pitch)
    self.assertEqual(librosa.note_to_midi('F4'), notes.notes[4].pitch)
    self.assertEqual(librosa.note_to_midi('D4'), notes.notes[5].pitch)
    self.assertEqual(librosa.note_to_midi('E4'), notes.notes[6].pitch)
    self.assertEqual(librosa.note_to_midi('C4'), notes.notes[7].pitch)

  def testBeams_sixteenthNotes(self):
    filename = os.path.join(resource_loader.get_data_files_path(),
                            'testdata/IMSLP00747-000.png')
    notes = self.engine.run([filename], output_notesequence=True)

    def _sixteenth_note(pitch, start_time):
      return music_pb2.NoteSequence.Note(
          pitch=librosa.note_to_midi(pitch),
          start_time=start_time,
          end_time=start_time + 0.25)

    # TODO(ringwalt): Fix the single quarter note detected before the treble
    # clef.
    # TODO(ringwalt): Detect the sixteenth rest that's missing before the first
    # real note.
    self.assertIn(_sixteenth_note('C4', 1.0), notes.notes)
    self.assertIn(_sixteenth_note('D4', 1.25), notes.notes)
    self.assertIn(_sixteenth_note('E4', 1.5), notes.notes)
    self.assertIn(_sixteenth_note('F4', 1.75), notes.notes)
    # TODO(ringwalt): The second D and E are detected with only one beam, even
    # though they are connected to the same beams as the F before them and the
    # C after them. Fix.

  def testIMSLP00747_000_structure_barlines(self):
    page = self.engine.run(
        os.path.join(resource_loader.get_data_files_path(),
                     'testdata/IMSLP00747-000.png')).page[0]
    self.assertEqual(len(page.system), 6)

    self.assertEqual(len(page.system[0].staff), 2)
    self.assertEqual(len(page.system[0].bar), 4)

    self.assertEqual(len(page.system[1].staff), 2)
    self.assertEqual(len(page.system[1].bar), 5)

    self.assertEqual(len(page.system[2].staff), 2)
    self.assertEqual(len(page.system[2].bar), 5)

    self.assertEqual(len(page.system[3].staff), 2)
    self.assertEqual(len(page.system[3].bar), 4)

    self.assertEqual(len(page.system[4].staff), 2)
    self.assertEqual(len(page.system[4].bar), 5)

    self.assertEqual(len(page.system[5].staff), 2)
    self.assertEqual(len(page.system[5].bar), 5)

    for system in page.system:
      for staff in system.staff:
        self.assertEqual(staff.staffline_distance, 16)

  def testMusicXML(self):
    filename = os.path.join(resource_loader.get_data_files_path(),
                            'testdata/IMSLP00747-000.png')
    score = self.engine.run([filename])
    num_measures = sum(
        len(system.bar) - 1 for page in score.page for system in page.system)
    musicxml = etree.fromstring(conversions.score_to_musicxml(score))
    self.assertEqual(2, len(musicxml.findall('part')))
    self.assertEqual(num_measures,
                     len(musicxml.find('part[1]').findall('measure')))

  def testProcessImage(self):
    pil_image = Image.open(
        os.path.join(resource_loader.get_data_files_path(),
                     'testdata/IMSLP00747-000.png')).convert('L')
    arr = np.array(pil_image.getdata(), np.uint8).reshape(
        # Size is (width, height).
        pil_image.size[1],
        pil_image.size[0])
    page = self.engine.process_image(arr)
    self.assertEqual(6, len(page.system))


if __name__ == '__main__':
  absltest.main()
