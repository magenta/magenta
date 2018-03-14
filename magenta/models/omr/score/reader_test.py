"""Tests for the OMR score reader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
from absl.testing import absltest
import librosa

from magenta.models.omr import conversions
from magenta.models.omr.protobuf import musicscore_pb2
from magenta.models.omr.score import reader
from magenta.protobuf import music_pb2

# pylint: disable=invalid-name
Glyph = musicscore_pb2.Glyph
Note = music_pb2.NoteSequence.Note
Point = musicscore_pb2.Point


class ReaderTest(absltest.TestCase):

  def testTreble_simple(self):
    staff = musicscore_pb2.Staff(
        staffline_distance=10,
        center_line=[Point(x=0, y=50), Point(x=100, y=50)],
        glyph=[
            Glyph(
                type=Glyph.CLEF_TREBLE,
                x=1,
                y_position=reader.TREBLE_CLEF_EXPECTED_Y),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=10, y_position=0),
        ])
    notes = conversions.page_to_notesequence(reader.ScoreReader().read_page(
        musicscore_pb2.Page(system=[musicscore_pb2.StaffSystem(staff=[staff])
                                   ])))
    self.assertEqual(
        notes,
        music_pb2.NoteSequence(notes=[
            Note(pitch=librosa.note_to_midi('B4'), start_time=0, end_time=1)
        ]))

  def testBass_simple(self):
    staff = musicscore_pb2.Staff(
        staffline_distance=10,
        center_line=[Point(x=0, y=50), Point(x=100, y=50)],
        glyph=[
            Glyph(
                type=Glyph.CLEF_BASS,
                x=1,
                y_position=reader.BASS_CLEF_EXPECTED_Y),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=10, y_position=0),
        ])
    notes = conversions.page_to_notesequence(reader.ScoreReader().read_page(
        musicscore_pb2.Page(system=[musicscore_pb2.StaffSystem(staff=[staff])
                                   ])))
    self.assertEqual(
        notes,
        music_pb2.NoteSequence(notes=[
            Note(pitch=librosa.note_to_midi('D3'), start_time=0, end_time=1)
        ]))

  def testTreble_accidentals(self):
    staff_1 = musicscore_pb2.Staff(
        staffline_distance=10,
        center_line=[Point(x=0, y=50), Point(x=100, y=50)],
        glyph=[
            Glyph(
                type=Glyph.CLEF_TREBLE,
                x=1,
                y_position=reader.TREBLE_CLEF_EXPECTED_Y),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=10, y_position=-6),
            Glyph(type=Glyph.FLAT, x=16, y_position=-4),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=20, y_position=-4),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=30, y_position=-2),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=40, y_position=-4),
        ])
    staff_2 = musicscore_pb2.Staff(
        staffline_distance=10,
        center_line=[Point(x=0, y=150), Point(x=100, y=150)],
        glyph=[
            Glyph(
                type=Glyph.CLEF_TREBLE,
                x=1,
                y_position=reader.TREBLE_CLEF_EXPECTED_Y),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=10, y_position=-6),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=20, y_position=-4),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=30, y_position=-2),
            Glyph(type=Glyph.SHARP, x=35, y_position=-2),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=40, y_position=-2),
            Glyph(type=Glyph.NATURAL, x=45, y_position=-2),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=50, y_position=-2),
        ])
    notes = conversions.page_to_notesequence(reader.ScoreReader().read_page(
        musicscore_pb2.Page(system=[
            musicscore_pb2.StaffSystem(staff=[staff_1]),
            musicscore_pb2.StaffSystem(staff=[staff_2])
        ])))
    self.assertEqual(
        notes,
        music_pb2.NoteSequence(notes=[
            # First staff.
            Note(pitch=librosa.note_to_midi('C4'), start_time=0, end_time=1),
            Note(pitch=librosa.note_to_midi('Eb4'), start_time=1, end_time=2),
            Note(pitch=librosa.note_to_midi('G4'), start_time=2, end_time=3),
            Note(pitch=librosa.note_to_midi('Eb4'), start_time=3, end_time=4),
            # Second staff.
            Note(pitch=librosa.note_to_midi('C4'), start_time=4, end_time=5),
            Note(pitch=librosa.note_to_midi('E4'), start_time=5, end_time=6),
            Note(pitch=librosa.note_to_midi('G4'), start_time=6, end_time=7),
            Note(pitch=librosa.note_to_midi('G#4'), start_time=7, end_time=8),
            Note(pitch=librosa.note_to_midi('G4'), start_time=8, end_time=9),
        ]))

  def testChords(self):
    stem_1 = musicscore_pb2.LineSegment(
        start=Point(x=20, y=10), end=Point(x=20, y=70))
    stem_2 = musicscore_pb2.LineSegment(
        start=Point(x=50, y=10), end=Point(x=50, y=70))
    staff = musicscore_pb2.Staff(
        staffline_distance=10,
        center_line=[Point(x=0, y=50), Point(x=100, y=50)],
        glyph=[
            Glyph(
                type=Glyph.CLEF_TREBLE,
                x=1,
                y_position=reader.TREBLE_CLEF_EXPECTED_Y),
            # Chord of 2 notes.
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=10, y_position=-4, stem=stem_1),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=10, y_position=-1, stem=stem_1),

            # Note not attached to a stem.
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=30, y_position=3),

            # Chord of 3 notes.
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=40, y_position=0, stem=stem_2),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=60, y_position=2, stem=stem_2),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=60, y_position=4, stem=stem_2),
        ])
    notes = conversions.page_to_notesequence(reader.ScoreReader().read_page(
        musicscore_pb2.Page(system=[musicscore_pb2.StaffSystem(staff=[staff])
                                   ])))
    self.assertEqual(
        notes,
        music_pb2.NoteSequence(notes=[
            # First chord.
            Note(pitch=librosa.note_to_midi('E4'), start_time=0, end_time=1),
            Note(pitch=librosa.note_to_midi('A4'), start_time=0, end_time=1),

            # Note without a stem.
            Note(pitch=librosa.note_to_midi('E5'), start_time=1, end_time=2),

            # Second chord.
            Note(pitch=librosa.note_to_midi('B4'), start_time=2, end_time=3),
            Note(pitch=librosa.note_to_midi('D5'), start_time=2, end_time=3),
            Note(pitch=librosa.note_to_midi('F5'), start_time=2, end_time=3),
        ]))

  def testBeams(self):
    beam_1 = musicscore_pb2.LineSegment(
        start=Point(x=10, y=20), end=Point(x=40, y=20))
    beam_2 = musicscore_pb2.LineSegment(
        start=Point(x=70, y=40), end=Point(x=90, y=40))
    beam_3 = musicscore_pb2.LineSegment(
        start=Point(x=70, y=60), end=Point(x=90, y=60))
    staff = musicscore_pb2.Staff(
        staffline_distance=10,
        center_line=[Point(x=0, y=50), Point(x=100, y=50)],
        glyph=[
            Glyph(
                type=Glyph.CLEF_TREBLE,
                x=1,
                y_position=reader.TREBLE_CLEF_EXPECTED_Y),
            # 2 eighth notes.
            Glyph(
                type=Glyph.NOTEHEAD_FILLED, x=10, y_position=-4, beam=[beam_1]),
            Glyph(
                type=Glyph.NOTEHEAD_FILLED, x=40, y_position=-1, beam=[beam_1]),
            # 1 quarter note.
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=50, y_position=0),
            # 2 sixteenth notes.
            Glyph(
                type=Glyph.NOTEHEAD_FILLED,
                x=60,
                y_position=-2,
                beam=[beam_2, beam_3]),
            Glyph(
                type=Glyph.NOTEHEAD_FILLED,
                x=90,
                y_position=2,
                beam=[beam_2, beam_3]),
        ])
    notes = conversions.page_to_notesequence(reader.ScoreReader().read_page(
        musicscore_pb2.Page(system=[musicscore_pb2.StaffSystem(staff=[staff])
                                   ])))
    self.assertEqual(
        notes,
        music_pb2.NoteSequence(notes=[
            Note(pitch=librosa.note_to_midi('E4'), start_time=0, end_time=0.5),
            Note(pitch=librosa.note_to_midi('A4'), start_time=0.5, end_time=1),
            Note(pitch=librosa.note_to_midi('B4'), start_time=1, end_time=2),
            Note(pitch=librosa.note_to_midi('G4'), start_time=2, end_time=2.25),
            Note(
                pitch=librosa.note_to_midi('D5'), start_time=2.25,
                end_time=2.5),
        ]))

  def testAllNoteheadTypes(self):
    staff = musicscore_pb2.Staff(
        staffline_distance=10,
        center_line=[Point(x=0, y=50), Point(x=100, y=50)],
        glyph=[
            Glyph(
                type=Glyph.CLEF_TREBLE,
                x=1,
                y_position=reader.TREBLE_CLEF_EXPECTED_Y),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=10, y_position=-6),
            Glyph(type=Glyph.NOTEHEAD_EMPTY, x=10, y_position=-6),
            Glyph(type=Glyph.NOTEHEAD_WHOLE, x=10, y_position=-6),
        ])
    notes = conversions.page_to_notesequence(reader.ScoreReader().read_page(
        musicscore_pb2.Page(system=[musicscore_pb2.StaffSystem(staff=[staff])
                                   ])))
    self.assertEqual(
        notes,
        music_pb2.NoteSequence(notes=[
            Note(pitch=librosa.note_to_midi('C4'), start_time=0, end_time=1),
            Note(pitch=librosa.note_to_midi('C4'), start_time=1, end_time=3),
            Note(pitch=librosa.note_to_midi('C4'), start_time=3, end_time=7),
        ]))

  def testStaffSystems(self):
    # 2 staff systems on separate pages, each with 2 staves, and no bars.
    system_1_staff_1 = musicscore_pb2.Staff(
        staffline_distance=10,
        center_line=[Point(x=0, y=50), Point(x=100, y=50)],
        glyph=[
            Glyph(
                type=Glyph.CLEF_TREBLE,
                x=1,
                y_position=reader.TREBLE_CLEF_EXPECTED_Y),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=10, y_position=-6),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=50, y_position=-2),
        ])
    system_1_staff_2 = musicscore_pb2.Staff(
        staffline_distance=10,
        center_line=[Point(x=0, y=150), Point(x=100, y=150)],
        glyph=[
            Glyph(
                type=Glyph.CLEF_BASS,
                x=2,
                y_position=reader.BASS_CLEF_EXPECTED_Y),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=10, y_position=0),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=40, y_position=2),
            # Played after the second note in the first staff, although it is to
            # the left of it.
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=45, y_position=4),
        ])
    system_2_staff_1 = musicscore_pb2.Staff(
        staffline_distance=10,
        center_line=[Point(x=0, y=250), Point(x=100, y=250)],
        glyph=[
            Glyph(
                type=Glyph.CLEF_TREBLE,
                x=1,
                y_position=reader.TREBLE_CLEF_EXPECTED_Y),
            Glyph(type=Glyph.REST_QUARTER, x=20, y_position=0),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=50, y_position=-2),
        ])
    system_2_staff_2 = musicscore_pb2.Staff(
        staffline_distance=10,
        center_line=[Point(x=0, y=250), Point(x=100, y=250)],
        glyph=[
            Glyph(
                type=Glyph.CLEF_BASS,
                x=2,
                y_position=reader.BASS_CLEF_EXPECTED_Y),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=10, y_position=0),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=40, y_position=2),
        ])
    notes = conversions.score_to_notesequence(reader.ScoreReader()(
        musicscore_pb2.Score(page=[
            musicscore_pb2.Page(system=[
                musicscore_pb2.StaffSystem(
                    staff=[system_1_staff_1, system_1_staff_2]),
            ]),
            musicscore_pb2.Page(system=[
                musicscore_pb2.StaffSystem(
                    staff=[system_2_staff_1, system_2_staff_2]),
            ]),
        ]),))
    self.assertEqual(
        notes,
        music_pb2.NoteSequence(notes=[
            # System 1, staff 1.
            Note(pitch=librosa.note_to_midi('C4'), start_time=0, end_time=1),
            Note(pitch=librosa.note_to_midi('G4'), start_time=1, end_time=2),
            # System 1, staff 2.
            Note(pitch=librosa.note_to_midi('D3'), start_time=0, end_time=1),
            Note(pitch=librosa.note_to_midi('F3'), start_time=1, end_time=2),
            Note(pitch=librosa.note_to_midi('A3'), start_time=2, end_time=3),
            # System 2, staff 1.
            # Quarter rest.
            Note(pitch=librosa.note_to_midi('G4'), start_time=4, end_time=5),
            # System 2, staff 2.
            Note(pitch=librosa.note_to_midi('D3'), start_time=3, end_time=4),
            Note(pitch=librosa.note_to_midi('F3'), start_time=4, end_time=5),
        ]))

  def testMeasures(self):
    # 2 staves in the same staff system with multiple bars.
    staff_1 = musicscore_pb2.Staff(
        staffline_distance=10,
        center_line=[Point(x=0, y=50), Point(x=300, y=50)],
        glyph=[
            Glyph(
                type=Glyph.CLEF_TREBLE,
                x=1,
                y_position=reader.TREBLE_CLEF_EXPECTED_Y),
            # Key signature.
            Glyph(type=Glyph.SHARP, x=10, y_position=+4),

            Glyph(type=Glyph.NOTEHEAD_FILLED, x=20, y_position=-2),

            # Accidental.
            Glyph(type=Glyph.FLAT, x=40, y_position=-1),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=50, y_position=-1),

            # Second bar.
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=120, y_position=0),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=180, y_position=+4),

            # Third bar.
            # Accidental not propagated to this note.
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=220, y_position=-1),
        ])
    staff_2 = musicscore_pb2.Staff(
        staffline_distance=10,
        center_line=[Point(x=0, y=150), Point(x=300, y=150)],
        glyph=[
            Glyph(
                type=Glyph.CLEF_BASS,
                x=1,
                y_position=reader.BASS_CLEF_EXPECTED_Y),
            # Key signature.
            Glyph(type=Glyph.FLAT, x=15, y_position=-2),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=20, y_position=-2),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=50, y_position=+2),

            # Second bar.
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=150, y_position=-2),

            # Third bar.
            Glyph(type=Glyph.REST_QUARTER, x=220, y_position=0),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=280, y_position=-2),
        ])
    staff_system = musicscore_pb2.StaffSystem(
        staff=[staff_1, staff_2],
        bar=[_bar(0), _bar(100), _bar(200),
             _bar(300)])
    notes = conversions.page_to_notesequence(reader.ScoreReader().read_page(
        musicscore_pb2.Page(system=[staff_system])))
    self.assertEqual(
        notes,
        music_pb2.NoteSequence(notes=[
            # Staff 1, bar 1.
            Note(pitch=librosa.note_to_midi('G4'), start_time=0, end_time=1),
            Note(pitch=librosa.note_to_midi('Ab4'), start_time=1, end_time=2),
            # Staff 1, bar 2.
            Note(pitch=librosa.note_to_midi('B4'), start_time=2, end_time=3),
            Note(pitch=librosa.note_to_midi('F#5'), start_time=3, end_time=4),
            # Staff 1, bar 3.
            Note(pitch=librosa.note_to_midi('A4'), start_time=4, end_time=5),
            # Staff 2, bar 1.
            Note(pitch=librosa.note_to_midi('Bb2'), start_time=0, end_time=1),
            Note(pitch=librosa.note_to_midi('F3'), start_time=1, end_time=2),
            # Staff 2, bar 2.
            Note(pitch=librosa.note_to_midi('Bb2'), start_time=2, end_time=3),
            # Staff 2, bar 3.
            Note(pitch=librosa.note_to_midi('Bb2'), start_time=5, end_time=6),
        ]))

  def testKeySignatures(self):
    # One staff per system, two systems.
    staff_1 = musicscore_pb2.Staff(
        glyph=[
            Glyph(type=Glyph.CLEF_TREBLE, x=5,
                  y_position=reader.TREBLE_CLEF_EXPECTED_Y),
            # D major key signature.
            Glyph(type=Glyph.SHARP, x=15, y_position=+4),
            Glyph(type=Glyph.SHARP, x=25, y_position=+1),

            # Accidental which cannot be interpreted as part of the key
            # signature.
            Glyph(type=Glyph.SHARP, x=35, y_position=+2),
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=45, y_position=+2),  # D#5

            Glyph(type=Glyph.NOTEHEAD_EMPTY, x=55, y_position=+1),  # C#5
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=65, y_position=-3),  # F#4

            # New measure. The key signature should be retained.
            Glyph(type=Glyph.NOTEHEAD_EMPTY, x=105, y_position=-3),  # F#4
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=125, y_position=+1),  # C#5
            # Accidental is not retained.
            Glyph(type=Glyph.NOTEHEAD_FILLED, x=145, y_position=+2),  # D5
        ])
    staff_2 = musicscore_pb2.Staff(
        glyph=[
            Glyph(type=Glyph.CLEF_TREBLE, x=5,
                  y_position=reader.TREBLE_CLEF_EXPECTED_Y),
            # No key signature on this line. No accidentals.
            Glyph(type=Glyph.NOTEHEAD_EMPTY, x=25, y_position=-3),  # F4
            Glyph(type=Glyph.NOTEHEAD_EMPTY, x=45, y_position=+1),  # C5
        ])
    notes = conversions.page_to_notesequence(reader.ScoreReader().read_page(
        musicscore_pb2.Page(system=[
            musicscore_pb2.StaffSystem(
                staff=[staff_1], bar=[_bar(0), _bar(100),
                                      _bar(200)]),
            musicscore_pb2.StaffSystem(staff=[staff_2]),
        ])))
    self.assertEqual(
        notes,
        music_pb2.NoteSequence(notes=[
            # First measure.
            Note(pitch=librosa.note_to_midi('D#5'), start_time=0, end_time=1),
            Note(pitch=librosa.note_to_midi('C#5'), start_time=1, end_time=3),
            Note(pitch=librosa.note_to_midi('F#4'), start_time=3, end_time=4),
            # Second measure.
            Note(pitch=librosa.note_to_midi('F#4'), start_time=4, end_time=6),
            Note(pitch=librosa.note_to_midi('C#5'), start_time=6, end_time=7),
            Note(pitch=librosa.note_to_midi('D5'), start_time=7, end_time=8),
            # Third measure on a new line, with no key signature.
            Note(pitch=librosa.note_to_midi('F4'), start_time=8, end_time=10),
            Note(pitch=librosa.note_to_midi('C5'), start_time=10, end_time=12),
        ]))


def _bar(x):
  return musicscore_pb2.StaffSystem.Bar(
      x=x, type=musicscore_pb2.StaffSystem.Bar.STANDARD_BAR)


if __name__ == '__main__':
  absltest.main()
