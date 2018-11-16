# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for MusicVAE data library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
import tensorflow as tf

from magenta.models.music_vae import data

import magenta.music as mm
from magenta.music import constants
from magenta.music import testing_lib
from magenta.protobuf import music_pb2

NO_EVENT = constants.MELODY_NO_EVENT
NOTE_OFF = constants.MELODY_NOTE_OFF
NO_DRUMS = 0
NO_CHORD = constants.NO_CHORD


def filter_instrument(sequence, instrument):
  filtered_sequence = music_pb2.NoteSequence()
  filtered_sequence.CopyFrom(sequence)
  del filtered_sequence.notes[:]
  filtered_sequence.notes.extend(
      [n for n in sequence.notes if n.instrument == instrument])
  return filtered_sequence


class NoteSequenceAugmenterTest(tf.test.TestCase):

  def setUp(self):
    sequence = music_pb2.NoteSequence()
    sequence.tempos.add(qpm=60)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(32, 100, 2, 4), (33, 100, 6, 11), (34, 100, 11, 13),
         (35, 100, 17, 18)])
    testing_lib.add_track_to_sequence(
        sequence, 1, [(57, 80, 4, 4.1), (58, 80, 12, 12.1)], is_drum=True)
    testing_lib.add_chords_to_sequence(
        sequence, [('N.C.', 0), ('C', 8), ('Am', 16)])
    self.sequence = sequence

  def testAugmentTranspose(self):
    augmenter = data.NoteSequenceAugmenter(transpose_range=(2, 2))
    augmented_sequence = augmenter.augment(self.sequence)

    expected_sequence = music_pb2.NoteSequence()
    expected_sequence.tempos.add(qpm=60)
    testing_lib.add_track_to_sequence(
        expected_sequence, 0,
        [(34, 100, 2, 4), (35, 100, 6, 11), (36, 100, 11, 13),
         (37, 100, 17, 18)])
    testing_lib.add_track_to_sequence(
        expected_sequence, 1, [(57, 80, 4, 4.1), (58, 80, 12, 12.1)],
        is_drum=True)
    testing_lib.add_chords_to_sequence(
        expected_sequence, [('N.C.', 0), ('D', 8), ('Bm', 16)])

    self.assertEqual(expected_sequence, augmented_sequence)

  def testAugmentStretch(self):
    augmenter = data.NoteSequenceAugmenter(stretch_range=(0.5, 0.5))
    augmented_sequence = augmenter.augment(self.sequence)

    expected_sequence = music_pb2.NoteSequence()
    expected_sequence.tempos.add(qpm=120)
    testing_lib.add_track_to_sequence(
        expected_sequence, 0,
        [(32, 100, 1, 2), (33, 100, 3, 5.5), (34, 100, 5.5, 6.5),
         (35, 100, 8.5, 9)])
    testing_lib.add_track_to_sequence(
        expected_sequence, 1, [(57, 80, 2, 2.05), (58, 80, 6, 6.05)],
        is_drum=True)
    testing_lib.add_chords_to_sequence(
        expected_sequence, [('N.C.', 0), ('C', 4), ('Am', 8)])

    self.assertEqual(expected_sequence, augmented_sequence)

  def testTfAugment(self):
    augmenter = data.NoteSequenceAugmenter(
        transpose_range=(-3, -3), stretch_range=(2.0, 2.0))

    with self.test_session() as sess:
      sequence_str = tf.placeholder(tf.string)
      augmented_sequence_str_ = augmenter.tf_augment(sequence_str)
      augmented_sequence_str = sess.run(
          [augmented_sequence_str_],
          feed_dict={sequence_str: self.sequence.SerializeToString()})
    augmented_sequence = music_pb2.NoteSequence.FromString(
        augmented_sequence_str[0])

    expected_sequence = music_pb2.NoteSequence()
    expected_sequence.tempos.add(qpm=30)
    testing_lib.add_track_to_sequence(
        expected_sequence, 0,
        [(29, 100, 4, 8), (30, 100, 12, 22), (31, 100, 22, 26),
         (32, 100, 34, 36)])
    testing_lib.add_track_to_sequence(
        expected_sequence, 1, [(57, 80, 8, 8.2), (58, 80, 24, 24.2)],
        is_drum=True)
    testing_lib.add_chords_to_sequence(
        expected_sequence, [('N.C.', 0), ('A', 16), ('Gbm', 32)])

    self.assertEqual(expected_sequence, augmented_sequence)


class BaseDataTest(object):

  def labels_to_inputs(self, labels, converter):
    return [data.np_onehot(l, converter.input_depth, converter.input_dtype)
            for l in labels]

  def assertArraySetsEqual(self, lhs, rhs):
    def _np_sorted(arr_list):
      return sorted(arr_list, key=lambda x: x.tostring())
    self.assertEqual(len(lhs), len(rhs))
    for a, b in zip(_np_sorted(lhs), _np_sorted(rhs)):
      # Convert bool type to int for easier-to-read error messages.
      if a.dtype == np.bool:
        a = a.astype(np.int)
      if b.dtype == np.bool:
        b = b.astype(np.int)
      np.testing.assert_array_equal(a, b)


class BaseOneHotDataTest(BaseDataTest):

  def testUnsliced(self):
    converter = self.converter_class(steps_per_quarter=1, slice_bars=None)
    tensors = converter.to_tensors(self.sequence)
    actual_unsliced_labels = [np.argmax(t, axis=-1) for t in tensors.outputs]
    self.assertArraySetsEqual(
        self.labels_to_inputs(self.expected_unsliced_labels, converter),
        tensors.inputs)
    self.assertArraySetsEqual(
        self.expected_unsliced_labels, actual_unsliced_labels)

  def testTfUnsliced(self):
    converter = self.converter_class(steps_per_quarter=1, slice_bars=None)
    with self.test_session() as sess:
      sequence = tf.placeholder(tf.string)
      input_tensors_, output_tensors_, _, lengths_ = converter.tf_to_tensors(
          sequence)
      input_tensors, output_tensors, lengths = sess.run(
          [input_tensors_, output_tensors_, lengths_],
          feed_dict={sequence: self.sequence.SerializeToString()})
    actual_input_tensors = [t[:l] for t, l in zip(input_tensors, lengths)]
    actual_unsliced_labels = [
        np.argmax(t, axis=-1)[:l] for t, l in zip(output_tensors, lengths)]

    self.assertArraySetsEqual(
        self.labels_to_inputs(self.expected_unsliced_labels, converter),
        actual_input_tensors)
    self.assertArraySetsEqual(
        self.expected_unsliced_labels, actual_unsliced_labels)

  def testUnslicedEndToken(self):
    orig_converter = self.converter_class(
        steps_per_quarter=1, slice_bars=None)
    self.assertEqual(None, orig_converter.end_token)
    converter = self.converter_class(
        steps_per_quarter=1, slice_bars=None, add_end_token=True)
    self.assertEqual(orig_converter.input_depth + 1, converter.input_depth)
    self.assertEqual(orig_converter.output_depth, converter.end_token)
    self.assertEqual(orig_converter.output_depth + 1, converter.output_depth)

    expected_unsliced_labels = [
        np.append(l, [converter.end_token])
        for l in self.expected_unsliced_labels]

    tensors = converter.to_tensors(self.sequence)
    actual_unsliced_labels = [np.argmax(t, axis=-1) for t in tensors.outputs]

    self.assertArraySetsEqual(
        self.labels_to_inputs(expected_unsliced_labels, converter),
        tensors.inputs)
    self.assertArraySetsEqual(expected_unsliced_labels, actual_unsliced_labels)

  def testSliced(self):
    converter = self.converter_class(
        steps_per_quarter=1, slice_bars=2, max_tensors_per_notesequence=None)
    tensors = converter.to_tensors(self.sequence)
    actual_sliced_labels = [np.argmax(t, axis=-1) for t in tensors.outputs]

    self.assertArraySetsEqual(
        self.labels_to_inputs(self.expected_sliced_labels, converter),
        tensors.inputs)
    self.assertArraySetsEqual(self.expected_sliced_labels, actual_sliced_labels)

  def testTfSliced(self):
    converter = self.converter_class(
        steps_per_quarter=1, slice_bars=2, max_tensors_per_notesequence=None)
    with self.test_session() as sess:
      sequence = tf.placeholder(tf.string)
      input_tensors_, output_tensors_, _, lengths_ = converter.tf_to_tensors(
          sequence)
      input_tensors, output_tensors, lengths = sess.run(
          [input_tensors_, output_tensors_, lengths_],
          feed_dict={sequence: self.sequence.SerializeToString()})
    actual_sliced_labels = [
        np.argmax(t, axis=-1)[:l] for t, l in zip(output_tensors, lengths)]

    self.assertArraySetsEqual(
        self.labels_to_inputs(self.expected_sliced_labels, converter),
        input_tensors)
    self.assertArraySetsEqual(self.expected_sliced_labels, actual_sliced_labels)


class BaseChordConditionedOneHotDataTest(BaseOneHotDataTest):

  def testUnslicedChordConditioned(self):
    converter = self.converter_class(
        steps_per_quarter=1, slice_bars=None,
        chord_encoding=mm.MajorMinorChordOneHotEncoding())
    tensors = converter.to_tensors(self.sequence)
    actual_unsliced_labels = [np.argmax(t, axis=-1) for t in tensors.outputs]
    actual_unsliced_chord_labels = [
        np.argmax(t, axis=-1) for t in tensors.controls]
    self.assertArraySetsEqual(
        self.labels_to_inputs(self.expected_unsliced_labels, converter),
        tensors.inputs)
    self.assertArraySetsEqual(
        self.expected_unsliced_labels, actual_unsliced_labels)
    self.assertArraySetsEqual(
        self.expected_unsliced_chord_labels, actual_unsliced_chord_labels)

  def testTfUnslicedChordConditioned(self):
    converter = self.converter_class(
        steps_per_quarter=1, slice_bars=None,
        chord_encoding=mm.MajorMinorChordOneHotEncoding())
    with self.test_session() as sess:
      sequence = tf.placeholder(tf.string)
      input_tensors_, output_tensors_, control_tensors_, lengths_ = (
          converter.tf_to_tensors(sequence))
      input_tensors, output_tensors, control_tensors, lengths = sess.run(
          [input_tensors_, output_tensors_, control_tensors_, lengths_],
          feed_dict={sequence: self.sequence.SerializeToString()})
    actual_input_tensors = [t[:l] for t, l in zip(input_tensors, lengths)]
    actual_unsliced_labels = [
        np.argmax(t, axis=-1)[:l] for t, l in zip(output_tensors, lengths)]
    actual_unsliced_chord_labels = [
        np.argmax(t, axis=-1)[:l] for t, l in zip(control_tensors, lengths)]

    self.assertArraySetsEqual(
        self.labels_to_inputs(self.expected_unsliced_labels, converter),
        actual_input_tensors)
    self.assertArraySetsEqual(
        self.expected_unsliced_labels, actual_unsliced_labels)
    self.assertArraySetsEqual(
        self.expected_unsliced_chord_labels, actual_unsliced_chord_labels)

  def testSlicedChordConditioned(self):
    converter = self.converter_class(
        steps_per_quarter=1, slice_bars=2, max_tensors_per_notesequence=None,
        chord_encoding=mm.MajorMinorChordOneHotEncoding())
    tensors = converter.to_tensors(self.sequence)
    actual_sliced_labels = [np.argmax(t, axis=-1) for t in tensors.outputs]
    actual_sliced_chord_labels = [
        np.argmax(t, axis=-1) for t in tensors.controls]

    self.assertArraySetsEqual(
        self.labels_to_inputs(self.expected_sliced_labels, converter),
        tensors.inputs)
    self.assertArraySetsEqual(self.expected_sliced_labels, actual_sliced_labels)
    self.assertArraySetsEqual(
        self.expected_sliced_chord_labels, actual_sliced_chord_labels)

  def testTfSlicedChordConditioned(self):
    converter = self.converter_class(
        steps_per_quarter=1, slice_bars=2, max_tensors_per_notesequence=None,
        chord_encoding=mm.MajorMinorChordOneHotEncoding())
    with self.test_session() as sess:
      sequence = tf.placeholder(tf.string)
      input_tensors_, output_tensors_, control_tensors_, lengths_ = (
          converter.tf_to_tensors(sequence))
      input_tensors, output_tensors, control_tensors, lengths = sess.run(
          [input_tensors_, output_tensors_, control_tensors_, lengths_],
          feed_dict={sequence: self.sequence.SerializeToString()})
    actual_sliced_labels = [
        np.argmax(t, axis=-1)[:l] for t, l in zip(output_tensors, lengths)]
    actual_sliced_chord_labels = [
        np.argmax(t, axis=-1)[:l] for t, l in zip(control_tensors, lengths)]

    self.assertArraySetsEqual(
        self.labels_to_inputs(self.expected_sliced_labels, converter),
        input_tensors)
    self.assertArraySetsEqual(self.expected_sliced_labels, actual_sliced_labels)
    self.assertArraySetsEqual(
        self.expected_sliced_chord_labels, actual_sliced_chord_labels)


class OneHotMelodyConverterTest(BaseChordConditionedOneHotDataTest,
                                tf.test.TestCase):

  def setUp(self):
    sequence = music_pb2.NoteSequence()
    sequence.tempos.add(qpm=60)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(32, 100, 2, 4), (33, 1, 6, 11), (34, 1, 11, 13),
         (35, 1, 17, 19)])
    testing_lib.add_track_to_sequence(
        sequence, 1,
        [(35, 127, 2, 4), (36, 50, 6, 8),
         (71, 100, 33, 37), (73, 100, 34, 37),
         (33, 1, 50, 55), (34, 1, 55, 56)])
    testing_lib.add_chords_to_sequence(
        sequence,
        [('F', 2), ('C', 8), ('Am', 16), ('N.C.', 20),
         ('Bb7', 32), ('G', 36), ('F', 48), ('C', 52)])
    self.sequence = sequence

    # Subtract min pitch (21).
    expected_unsliced_events = [
        (NO_EVENT, NO_EVENT, 11, NO_EVENT,
         NOTE_OFF, NO_EVENT, 12, NO_EVENT,
         NO_EVENT, NO_EVENT, NO_EVENT, 13,
         NO_EVENT, NOTE_OFF, NO_EVENT, NO_EVENT),
        (NO_EVENT, 14, NO_EVENT, NOTE_OFF),
        (NO_EVENT, NO_EVENT, 14, NO_EVENT,
         NOTE_OFF, NO_EVENT, 15, NO_EVENT),
        (NO_EVENT, 50, 52, NO_EVENT,
         NO_EVENT, NOTE_OFF, NO_EVENT, NO_EVENT),
        (NO_EVENT, NO_EVENT, 12, NO_EVENT,
         NO_EVENT, NO_EVENT, NO_EVENT, 13),
    ]
    self.expected_unsliced_labels = [
        np.array(es) + 2 for es in expected_unsliced_events]

    expected_sliced_events = [
        (NO_EVENT, NO_EVENT, 11, NO_EVENT,
         NOTE_OFF, NO_EVENT, 12, NO_EVENT),
        (NO_EVENT, NO_EVENT, 12, NO_EVENT,
         NO_EVENT, NO_EVENT, NO_EVENT, 13),
        (NO_EVENT, NO_EVENT, NO_EVENT, 13,
         NO_EVENT, NOTE_OFF, NO_EVENT, NO_EVENT),
        (NO_EVENT, NO_EVENT, 14, NO_EVENT,
         NOTE_OFF, NO_EVENT, 15, NO_EVENT),
        (NO_EVENT, 50, 52, NO_EVENT,
         NO_EVENT, NOTE_OFF, NO_EVENT, NO_EVENT)
    ]
    self.expected_sliced_labels = [
        np.array(es) + 2 for es in expected_sliced_events]

    chord_encoding = mm.MajorMinorChordOneHotEncoding()

    expected_unsliced_chord_events = [
        (NO_CHORD, NO_CHORD, 'F', 'F',
         'F', 'F', 'F', 'F',
         'C', 'C', 'C', 'C',
         'C', 'C', 'C', 'C'),
        ('Am', 'Am', 'Am', 'Am'),
        (NO_CHORD, NO_CHORD, 'F', 'F',
         'F', 'F', 'F', 'F'),
        ('Bb7', 'Bb7', 'Bb7', 'Bb7',
         'G', 'G', 'G', 'G'),
        ('F', 'F', 'F', 'F',
         'C', 'C', 'C', 'C'),
    ]
    self.expected_unsliced_chord_labels = [
        np.array([chord_encoding.encode_event(e) for e in es])
        for es in expected_unsliced_chord_events]

    expected_sliced_chord_events = [
        (NO_CHORD, NO_CHORD, 'F', 'F',
         'F', 'F', 'F', 'F'),
        ('F', 'F', 'F', 'F',
         'C', 'C', 'C', 'C'),
        ('C', 'C', 'C', 'C',
         'C', 'C', 'C', 'C'),
        (NO_CHORD, NO_CHORD, 'F', 'F',
         'F', 'F', 'F', 'F'),
        ('Bb7', 'Bb7', 'Bb7', 'Bb7',
         'G', 'G', 'G', 'G'),
    ]
    self.expected_sliced_chord_labels = [
        np.array([chord_encoding.encode_event(e) for e in es])
        for es in expected_sliced_chord_events]

    self.converter_class = data.OneHotMelodyConverter

  def testMaxOutputsPerNoteSequence(self):
    converter = data.OneHotMelodyConverter(
        steps_per_quarter=1, slice_bars=2, max_tensors_per_notesequence=2)
    self.assertEqual(2, len(converter.to_tensors(self.sequence).inputs))

    converter.max_tensors_per_notesequence = 3
    self.assertEqual(3, len(converter.to_tensors(self.sequence).inputs))

    converter.max_tensors_per_notesequence = 100
    self.assertEqual(5, len(converter.to_tensors(self.sequence).inputs))

  def testIsTraining(self):
    converter = data.OneHotMelodyConverter(
        steps_per_quarter=1, slice_bars=2, max_tensors_per_notesequence=2)
    self.is_training = True
    self.assertEqual(2, len(converter.to_tensors(self.sequence).inputs))

    converter.max_tensors_per_notesequence = None
    self.assertEqual(5, len(converter.to_tensors(self.sequence).inputs))

  def testToNoteSequence(self):
    converter = data.OneHotMelodyConverter(
        steps_per_quarter=1, slice_bars=4, max_tensors_per_notesequence=1)
    tensors = converter.to_tensors(
        filter_instrument(self.sequence, 0))
    sequences = converter.to_notesequences(tensors.outputs)

    self.assertEqual(1, len(sequences))
    expected_sequence = music_pb2.NoteSequence(ticks_per_quarter=220)
    expected_sequence.tempos.add(qpm=120)
    testing_lib.add_track_to_sequence(
        expected_sequence, 0,
        [(32, 80, 1.0, 2.0), (33, 80, 3.0, 5.5), (34, 80, 5.5, 6.5)])
    self.assertProtoEquals(expected_sequence, sequences[0])

  def testToNoteSequenceChordConditioned(self):
    converter = data.OneHotMelodyConverter(
        steps_per_quarter=1, slice_bars=4, max_tensors_per_notesequence=1,
        chord_encoding=mm.MajorMinorChordOneHotEncoding())
    tensors = converter.to_tensors(
        filter_instrument(self.sequence, 0))
    sequences = converter.to_notesequences(tensors.outputs, tensors.controls)

    self.assertEqual(1, len(sequences))
    expected_sequence = music_pb2.NoteSequence(ticks_per_quarter=220)
    expected_sequence.tempos.add(qpm=120)
    testing_lib.add_track_to_sequence(
        expected_sequence, 0,
        [(32, 80, 1.0, 2.0), (33, 80, 3.0, 5.5), (34, 80, 5.5, 6.5)])
    testing_lib.add_chords_to_sequence(
        expected_sequence, [('N.C.', 0), ('F', 1), ('C', 4)])
    self.assertProtoEquals(expected_sequence, sequences[0])


class OneHotDrumsConverterTest(BaseOneHotDataTest, tf.test.TestCase):

  def setUp(self):
    sequence = music_pb2.NoteSequence()
    sequence.tempos.add(qpm=60)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(35, 100, 0, 10), (44, 55, 1, 2), (40, 45, 4, 5), (35, 45, 9, 10),
         (40, 45, 13, 13), (55, 120, 16, 18), (60, 100, 16, 17),
         (52, 99, 19, 20)],
        is_drum=True)
    testing_lib.add_track_to_sequence(
        sequence, 1,
        [(35, 55, 1, 2), (40, 45, 25, 26), (55, 120, 28, 30), (60, 100, 28, 29),
         (52, 99, 31, 33)],
        is_drum=True)
    self.sequence = sequence

    expected_unsliced_events = [
        (1, 5, NO_DRUMS, NO_DRUMS,
         2, NO_DRUMS, NO_DRUMS, NO_DRUMS),
        (NO_DRUMS, 1, NO_DRUMS, NO_DRUMS,
         NO_DRUMS, 2, NO_DRUMS, NO_DRUMS,
         160, NO_DRUMS, NO_DRUMS, 256),
        (NO_DRUMS, 2, NO_DRUMS, NO_DRUMS,
         160, NO_DRUMS, NO_DRUMS, 256)
    ]
    self.expected_unsliced_labels = [
        np.array(es) for es in expected_unsliced_events]

    expected_sliced_events = [
        (1, 5, NO_DRUMS, NO_DRUMS,
         2, NO_DRUMS, NO_DRUMS, NO_DRUMS),
        (NO_DRUMS, 1, NO_DRUMS, NO_DRUMS,
         NO_DRUMS, 2, NO_DRUMS, NO_DRUMS),
        (NO_DRUMS, 2, NO_DRUMS, NO_DRUMS,
         160, NO_DRUMS, NO_DRUMS, 256)
    ]
    self.expected_sliced_labels = [
        np.array(es) for es in expected_sliced_events]

    self.converter_class = data.DrumsConverter

  def testMaxOutputsPerNoteSequence(self):
    converter = data.DrumsConverter(
        steps_per_quarter=1, slice_bars=1, max_tensors_per_notesequence=2)
    self.assertEqual(2, len(converter.to_tensors(self.sequence).inputs))

    converter.max_tensors_per_notesequence = 3
    self.assertEqual(3, len(converter.to_tensors(self.sequence).inputs))

    converter.max_tensors_per_notesequence = 100
    self.assertEqual(5, len(converter.to_tensors(self.sequence).inputs))

  def testIsTraining(self):
    converter = data.DrumsConverter(
        steps_per_quarter=1, slice_bars=1, max_tensors_per_notesequence=2)
    self.is_training = True
    self.assertEqual(2, len(converter.to_tensors(self.sequence).inputs))

    converter.max_tensors_per_notesequence = None
    self.assertEqual(5, len(converter.to_tensors(self.sequence).inputs))

  def testToNoteSequence(self):
    converter = data.DrumsConverter(
        steps_per_quarter=1, slice_bars=2, max_tensors_per_notesequence=1)
    tensors = converter.to_tensors(
        filter_instrument(self.sequence, 1))
    sequences = converter.to_notesequences(tensors.outputs)

    self.assertEqual(1, len(sequences))
    expected_sequence = music_pb2.NoteSequence(ticks_per_quarter=220)
    expected_sequence.tempos.add(qpm=120)
    testing_lib.add_track_to_sequence(
        expected_sequence, 9,
        [(38, 80, 0.5, 1.0),
         (48, 80, 2.0, 2.5), (49, 80, 2.0, 2.5),
         (51, 80, 3.5, 4.0)],
        is_drum=True)
    self.assertProtoEquals(expected_sequence, sequences[0])


class RollInputsOneHotDrumsConverterTest(OneHotDrumsConverterTest):

  def labels_to_inputs(self, labels, converter):
    inputs = []
    for label_arr in labels:
      input_ = np.zeros((len(label_arr), converter.input_depth),
                        converter.input_dtype)
      for i, l in enumerate(label_arr):
        if l == converter.end_token:
          input_[i, -2] = 1
        elif l == 0:
          input_[i, -1] = 1
        else:
          j = 0
          while l:
            input_[i, j] = l % 2
            l >>= 1
            j += 1
          assert np.any(input_[i]), label_arr.astype(np.int)
      inputs.append(input_)
    return inputs

  def setUp(self):
    super(RollInputsOneHotDrumsConverterTest, self).setUp()
    self.converter_class = functools.partial(
        data.DrumsConverter, roll_input=True)


class RollOutputsDrumsConverterTest(BaseDataTest, tf.test.TestCase):

  def setUp(self):
    sequence = music_pb2.NoteSequence()
    sequence.tempos.add(qpm=60)
    testing_lib.add_track_to_sequence(
        sequence, 0,
        [(35, 100, 0, 10), (35, 55, 1, 2), (44, 55, 1, 2),
         (40, 45, 4, 5),
         (35, 45, 9, 10),
         (40, 45, 13, 13),
         (55, 120, 16, 18), (60, 100, 16, 17), (52, 99, 19, 20),
         (40, 45, 33, 34), (55, 120, 36, 37), (60, 100, 36, 37),
         (52, 99, 39, 42)],
        is_drum=True)
    testing_lib.add_track_to_sequence(
        sequence, 1,
        [(35, 100, 5, 10), (35, 55, 6, 8), (44, 55, 7, 9)],
        is_drum=False)
    self.sequence = sequence

  def testSliced(self):
    expected_sliced_events = [
        ([0], [0, 2], [], [],
         [1], [], [], []),
        ([], [0], [], [],
         [], [1], [], []),
        ([], [1], [], [],
         [5, 7], [], [], [8]),
    ]
    expected_silent_array = np.array([
        [0, 0, 1, 1, 0, 1, 1, 1],
        [1, 0, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 0, 1, 1, 0],
    ])
    expected_output_tensors = np.zeros(
        (len(expected_sliced_events), 8, len(data.REDUCED_DRUM_PITCH_CLASSES)),
        np.bool)
    for i, events in enumerate(expected_sliced_events):
      for j, e in enumerate(events):
        expected_output_tensors[i, j, e] = 1

    converter = data.DrumsConverter(
        pitch_classes=data.REDUCED_DRUM_PITCH_CLASSES,
        slice_bars=2,
        steps_per_quarter=1,
        roll_input=True,
        roll_output=True,
        max_tensors_per_notesequence=None)

    self.assertEqual(10, converter.input_depth)
    self.assertEqual(9, converter.output_depth)

    tensors = converter.to_tensors(self.sequence)

    self.assertArraySetsEqual(
        np.append(
            expected_output_tensors,
            np.expand_dims(expected_silent_array, axis=2),
            axis=2),
        tensors.inputs)
    self.assertArraySetsEqual(expected_output_tensors, tensors.outputs)

  def testToNoteSequence(self):
    converter = data.DrumsConverter(
        pitch_classes=data.REDUCED_DRUM_PITCH_CLASSES,
        slice_bars=None,
        gap_bars=None,
        steps_per_quarter=1,
        roll_input=True,
        roll_output=True,
        max_tensors_per_notesequence=None)

    tensors = converter.to_tensors(self.sequence)
    sequences = converter.to_notesequences(tensors.outputs)

    self.assertEqual(1, len(sequences))
    expected_sequence = music_pb2.NoteSequence(ticks_per_quarter=220)
    expected_sequence.tempos.add(qpm=120)
    testing_lib.add_track_to_sequence(
        expected_sequence, 0,
        [(36, 80, 0, 0.5), (42, 80, 0.5, 1.0), (36, 80, 0.5, 1.0),
         (38, 80, 2.0, 2.5),
         (36, 80, 4.5, 5.0),
         (38, 80, 6.5, 7.0),
         (48, 80, 8.0, 8.5), (49, 80, 8.0, 8.5), (51, 80, 9.5, 10.0),
         (38, 80, 16.5, 17.0), (48, 80, 18.0, 18.5), (49, 80, 18.0, 18.5),
         (51, 80, 19.5, 20.0)],
        is_drum=True)
    for n in expected_sequence.notes:
      n.instrument = 9
    self.assertProtoEquals(expected_sequence, sequences[0])


class TrioConverterTest(BaseDataTest, tf.test.TestCase):

  def setUp(self):
    sequence = music_pb2.NoteSequence()
    sequence.tempos.add(qpm=60)
    # Mel 1, coverage bars: [3, 9] / [2, 9]
    testing_lib.add_track_to_sequence(
        sequence, 1, [(51, 1, 13, 37)])
    # Mel 2, coverage bars: [1, 3] / [0, 4]
    testing_lib.add_track_to_sequence(
        sequence, 2, [(52, 1, 4, 16)])
    # Bass, coverage bars: [0, 1], [4, 6] / [0, 7]
    testing_lib.add_track_to_sequence(
        sequence, 3, [(50, 1, 2, 5), (49, 1, 16, 25)])
    # Drum, coverage bars: [0, 2], [6, 7] / [0, 3], [5, 8]
    testing_lib.add_track_to_sequence(
        sequence, 4,
        [(35, 1, 0, 1), (40, 1, 4, 5),
         (35, 1, 9, 9), (35, 1, 25, 25),
         (40, 1, 29, 29)],
        is_drum=True)
    # Chords.
    testing_lib.add_chords_to_sequence(
        sequence, [('C', 4), ('Am', 16), ('G', 32)])

    for n in sequence.notes:
      if n.instrument == 1:
        n.program = 0
      elif n.instrument == 2:
        n.program = 10
      elif n.instrument == 3:
        n.program = 33

    self.sequence = sequence

    m1 = np.array(
        [NO_EVENT] * 13 + [30] + [NO_EVENT] * 23 + [NOTE_OFF] + [NO_EVENT] * 2,
        np.int32) + 2
    m2 = np.array(
        [NO_EVENT] * 4 + [31] + [NO_EVENT] * 11 + [NOTE_OFF] + [NO_EVENT] * 23,
        np.int32) + 2
    b = np.array(
        [NO_EVENT, NO_EVENT, 29, NO_EVENT, NO_EVENT, NOTE_OFF] +
        [NO_EVENT] * 10 + [28] + [NO_EVENT] * 8 + [NOTE_OFF] + [NO_EVENT] * 14,
        np.int32) + 2
    d = ([1, NO_DRUMS, NO_DRUMS, NO_DRUMS,
          2, NO_DRUMS, NO_DRUMS, NO_DRUMS,
          NO_DRUMS, 1, NO_DRUMS, NO_DRUMS] +
         [NO_DRUMS] * 12 +
         [NO_DRUMS, 1, NO_DRUMS, NO_DRUMS,
          NO_DRUMS, 2, NO_DRUMS, NO_DRUMS] +
         [NO_DRUMS] * 4)

    c = [NO_CHORD, NO_CHORD, NO_CHORD, NO_CHORD,
         'C', 'C', 'C', 'C',
         'C', 'C', 'C', 'C',
         'C', 'C', 'C', 'C',
         'Am', 'Am', 'Am', 'Am',
         'Am', 'Am', 'Am', 'Am',
         'Am', 'Am', 'Am', 'Am',
         'Am', 'Am', 'Am', 'Am',
         'G', 'G', 'G', 'G']

    expected_sliced_sets = [
        ((2, 4), (m1, b, d)),
        ((5, 7), (m1, b, d)),
        ((6, 8), (m1, b, d)),
        ((0, 2), (m2, b, d)),
        ((1, 3), (m2, b, d)),
        ((2, 4), (m2, b, d)),
    ]

    self.expected_sliced_labels = [
        np.stack([l[i*4:j*4] for l in x]) for (i, j), x in expected_sliced_sets]

    chord_encoding = mm.MajorMinorChordOneHotEncoding()
    expected_sliced_chord_events = [
        c[i*4:j*4] for (i, j), _ in expected_sliced_sets]
    self.expected_sliced_chord_labels = [
        np.array([chord_encoding.encode_event(e) for e in es])
        for es in expected_sliced_chord_events]

  def testSliced(self):
    converter = data.TrioConverter(
        steps_per_quarter=1, gap_bars=1, slice_bars=2,
        max_tensors_per_notesequence=None)
    tensors = converter.to_tensors(self.sequence)
    self.assertArraySetsEqual(tensors.inputs, tensors.outputs)
    actual_sliced_labels = [
        np.stack(np.argmax(s, axis=-1) for s in np.split(t, [90, 180], axis=-1))
        for t in tensors.outputs]

    self.assertArraySetsEqual(self.expected_sliced_labels, actual_sliced_labels)

  def testSlicedChordConditioned(self):
    converter = data.TrioConverter(
        steps_per_quarter=1, gap_bars=1, slice_bars=2,
        max_tensors_per_notesequence=None,
        chord_encoding=mm.MajorMinorChordOneHotEncoding())
    tensors = converter.to_tensors(self.sequence)
    self.assertArraySetsEqual(tensors.inputs, tensors.outputs)
    actual_sliced_labels = [
        np.stack(np.argmax(s, axis=-1) for s in np.split(t, [90, 180], axis=-1))
        for t in tensors.outputs]
    actual_sliced_chord_labels = [
        np.argmax(t, axis=-1) for t in tensors.controls]

    self.assertArraySetsEqual(self.expected_sliced_labels, actual_sliced_labels)
    self.assertArraySetsEqual(
        self.expected_sliced_chord_labels, actual_sliced_chord_labels)

  def testToNoteSequence(self):
    converter = data.TrioConverter(
        steps_per_quarter=1, slice_bars=2, max_tensors_per_notesequence=1)

    mel_oh = data.np_onehot(self.expected_sliced_labels[3][0], 90)
    bass_oh = data.np_onehot(self.expected_sliced_labels[3][1], 90)
    drums_oh = data.np_onehot(self.expected_sliced_labels[3][2], 512)
    output_tensors = np.concatenate([mel_oh, bass_oh, drums_oh], axis=-1)

    sequences = converter.to_notesequences([output_tensors])
    self.assertEqual(1, len(sequences))

    self.assertProtoEquals(
        """
        ticks_per_quarter: 220
        tempos < qpm: 120 >
        notes <
          instrument: 0 pitch: 52 start_time: 2.0 end_time: 4.0 program: 0
          velocity: 80
        >
        notes <
          instrument: 1 pitch: 50 start_time: 1.0 end_time: 2.5 program: 33
          velocity: 80
        >
        notes <
          instrument: 9 pitch: 36 start_time: 0.0 end_time: 0.5 velocity: 80
          is_drum: True
        >
        notes <
          instrument: 9 pitch: 38 start_time: 2.0 end_time: 2.5 velocity: 80
          is_drum: True
        >
        total_time: 4.0
        """,
        sequences[0])

  def testToNoteSequenceChordConditioned(self):
    converter = data.TrioConverter(
        steps_per_quarter=1, slice_bars=2, max_tensors_per_notesequence=1,
        chord_encoding=mm.MajorMinorChordOneHotEncoding())

    mel_oh = data.np_onehot(self.expected_sliced_labels[3][0], 90)
    bass_oh = data.np_onehot(self.expected_sliced_labels[3][1], 90)
    drums_oh = data.np_onehot(self.expected_sliced_labels[3][2], 512)
    chords_oh = data.np_onehot(self.expected_sliced_chord_labels[3], 25)

    output_tensors = np.concatenate([mel_oh, bass_oh, drums_oh], axis=-1)

    sequences = converter.to_notesequences([output_tensors], [chords_oh])
    self.assertEqual(1, len(sequences))

    self.assertProtoEquals(
        """
        ticks_per_quarter: 220
        tempos < qpm: 120 >
        notes <
          instrument: 0 pitch: 52 start_time: 2.0 end_time: 4.0 program: 0
          velocity: 80
        >
        notes <
          instrument: 1 pitch: 50 start_time: 1.0 end_time: 2.5 program: 33
          velocity: 80
        >
        notes <
          instrument: 9 pitch: 36 start_time: 0.0 end_time: 0.5 velocity: 80
          is_drum: True
        >
        notes <
          instrument: 9 pitch: 38 start_time: 2.0 end_time: 2.5 velocity: 80
          is_drum: True
        >
        text_annotations <
          text: 'N.C.' annotation_type: CHORD_SYMBOL
        >
        text_annotations <
          time: 2.0 text: 'C' annotation_type: CHORD_SYMBOL
        >
        total_time: 4.0
        """,
        sequences[0])


if __name__ == '__main__':
  tf.test.main()
