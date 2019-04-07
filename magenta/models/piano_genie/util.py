# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for Piano Genie."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pretty_midi
import tensorflow as tf

_PITCHCLASSES = [
    'C', 'C#', 'D', 'Eb', 'E', 'F',
    'F#', 'G', 'Ab', 'A', 'Bb', 'B']
_PITCHCLASSES_SET = set(_PITCHCLASSES)
_CHORDFAMILIES = ['', 'm', '+', 'dim', '7', 'maj7', 'm7', 'm7b5']
_CHORDFAMILIES_SET = set(_CHORDFAMILIES)
_CHORDFAMILY_TO_INTERVALS = {
    '': [0, 4, 7],
    'm': [0, 3, 7],
    '+': [0, 4, 8],
    'dim': [0, 3, 6],
    '7': [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'm7': [0, 3, 7, 10],
    'm7b5': [0, 3, 6, 10],
}

_ID_TO_PITCHCLASS = _PITCHCLASSES
_PITCHCLASS_TO_ID = {p:i for i, p in enumerate(_ID_TO_PITCHCLASS)}

NO_KEYSIG_SYMBOL = 'N.K.'
_ID_TO_KEYSIG = [NO_KEYSIG_SYMBOL] + _PITCHCLASSES
_KEYSIG_TO_ID = {k:i for i, k in enumerate(_ID_TO_KEYSIG)}
NUM_KEYSIGS = len(_ID_TO_KEYSIG)

NO_CHORD_SYMBOL = 'N.C.'
_ID_TO_CHORD = [NO_CHORD_SYMBOL]
for pc in _PITCHCLASSES:
  for cf in _CHORDFAMILIES:
    _ID_TO_CHORD.append(pc + cf)
_CHORD_TO_ID = {c:i for i, c in enumerate(_ID_TO_CHORD)}
NUM_CHORDS = len(_ID_TO_CHORD)

NO_CHORDROOT_SYMBOL = 'N.CR.'
_ID_TO_CHORDROOT = [NO_CHORDROOT_SYMBOL] + _PITCHCLASSES
_CHORDROOT_TO_ID = {c:i for i, c in enumerate(_ID_TO_CHORDROOT)}
NUM_CHORDROOTS = len(_ID_TO_CHORDROOT)

NO_CHORDFAMILY_SYMBOL = 'N.CF.'
_ID_TO_CHORDFAMILY = [NO_CHORDFAMILY_SYMBOL] + _CHORDFAMILIES
_CHORDFAMILY_TO_ID = {c:i for i, c in enumerate(_ID_TO_CHORDFAMILY)}
NUM_CHORDFAMILIES = len(_ID_TO_CHORDFAMILY)


def demidify(pitches):
  """Transforms MIDI pitches [21,108] to [0, 88)."""
  assertions = [
      tf.assert_greater_equal(pitches, 21),
      tf.assert_less_equal(pitches, 108)
  ]
  with tf.control_dependencies(assertions):
    return pitches - 21


def remidify(pitches):
  """Transforms [0, 88) to MIDI pitches [21, 108]."""
  assertions = [
      tf.assert_greater_equal(pitches, 0),
      tf.assert_less_equal(pitches, 87)
  ]
  with tf.control_dependencies(assertions):
    return pitches + 21


def id_to_pitchclass(i):
  """Translates integer to pitchclass e.g. 1->'C#'"""
  if i < 0 or i >= len(_ID_TO_PITCHCLASS):
    raise ValueError('Invalid pitchclass ID specified')
  return _ID_TO_PITCHCLASS[i]


def pitchclass_to_id(k):
  """Translates pitchclass to integer e.g. 'C#'->1"""
  if k not in _PITCHCLASS_TO_ID:
    raise ValueError('Invalid pitchclass specified')
  return _PITCHCLASS_TO_ID[k]


def id_to_keysig(i):
  """Translates integer to keysig e.g. 1->'C'"""
  if i < 0 or i >= len(_ID_TO_KEYSIG):
    raise ValueError('Invalid keysig ID specified')
  return _ID_TO_KEYSIG[i]


def keysig_to_id(k):
  """Translates keysig to integer e.g. 'C'->1"""
  if k not in _KEYSIG_TO_ID:
    raise ValueError('Invalid keysig specified')
  return _KEYSIG_TO_ID[k]


def id_to_chord(i):
  """Translates integer to chord e.g. 2->'Cm'"""
  if i < 0 or i >= len(_ID_TO_CHORD):
    raise ValueError('Invalid chord ID specified')
  return _ID_TO_CHORD[i]


def chord_to_id(c):
  """Translates chord to integer e.g. 'Cm'->2"""
  if c not in _CHORD_TO_ID:
    raise ValueError('Invalid chord specified')
  return _CHORD_TO_ID[c]


def id_to_chordroot(i):
  """Translates integer to chordroot e.g. 1->'C'"""
  if i < 0 or i >= len(_ID_TO_CHORDROOT):
    raise ValueError('Invalid chordroot ID specified')
  return _ID_TO_CHORDROOT[i]


def chordroot_to_id(c):
  """Translates chordroot to integer e.g. 'C'->1"""
  if c not in _CHORDROOT_TO_ID:
    raise ValueError('Invalid chordroot specified')
  return _CHORDROOT_TO_ID[c]


def id_to_chordfamily(i):
  """Translates integer to chordfamily e.g. 2->'m'"""
  if i < 0 or i >= len(_ID_TO_CHORDFAMILY):
    raise ValueError('Invalid chordfamily ID specified')
  return _ID_TO_CHORDFAMILY[i]


def chordfamily_to_id(c):
  """Translates chordfamily to integer e.g. 'm'->2"""
  if c not in _CHORDFAMILY_TO_ID:
    raise ValueError('Invalid chordfamily specified')
  return _CHORDFAMILY_TO_ID[c]


def chord_split(c):
  """Splits chord into root and family e.g. 'Cm7b5'->'C','m7b5'"""
  if c not in _CHORD_TO_ID:
    raise ValueError('Invalid chord specified')

  if c == 'N.C.':
    return None, None

  if len(c) == 1:
    cr = c[0]
    cf = ''
  else:
    if c[1] == 'b' or c[1] == '#':
      cr = c[:2]
      cf = c[2:]
    else:
      cr = c[0]
      cf = c[1:]

  assert cr in _PITCHCLASSES_SET
  assert cf in _CHORDFAMILIES_SET

  return cr, cf


def align_note_times_to_change_times(note_times, change_times, tol=0.):
  """Finds relevant chord change indices for a list of timestamps"""
  # Tolerance handles notes that start imperceptibly before the next chord
  if not np.all(np.diff(change_times) > 0):
    raise ValueError('Change times must be strictly increasing')
  if change_times[0] != 0:
    raise ValueError('First change must at time 0.')
  if tol < 0:
    raise ValueError('Tolerance must be >= 0.')
  if type(change_times) == list:
    change_times = np.array(change_times)
  idxs = np.searchsorted(change_times - tol, note_times, side='right') - 1
  return idxs


def notes_to_prev_audio(
    midi_pitches,
    start_times,
    end_times,
    prevlen=10.,
    fs=44100):
  """Creates a preview waveform from piano notes. Requires pyfluidsynth."""
  midi = pretty_midi.PrettyMIDI()

  piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
  piano = pretty_midi.Instrument(program=piano_program)

  offset = start_times[0]
  start_times = np.copy(start_times) - offset
  end_times = np.copy(end_times) - offset

  for p, s, e in zip(midi_pitches, start_times, end_times):
    if prevlen is not None and s > prevlen:
      break
    note = pretty_midi.Note(velocity=90, pitch=p, start=s, end=e)
    piano.notes.append(note)

  midi.instruments.append(piano)

  wav = midi.fluidsynth(fs=fs)
  slen = int(prevlen * fs)
  if prevlen is not None:
    wav = wav[:slen]
    wav = np.pad(wav, [[0, slen - wav.shape[0]]], 'constant')

  return wav.astype(np.float32)


def chords_to_prev_audio(
    start_times,
    chords,
    base_pitch=48,
    prevlen=10.,
    fs=44100):
  """Creates a preview waveform from chords. Requires pyfluidsynth."""
  midi = pretty_midi.PrettyMIDI()

  cello_program = pretty_midi.instrument_name_to_program('Cello')
  cello = pretty_midi.Instrument(program=cello_program)

  offset = start_times[0]
  start_times = np.copy(start_times) - offset

  chords = [chord_split(id_to_chord(i)) for i in chords]
  for i in range(len(chords) - 1):
    s = start_times[i]
    e = start_times[i+1]
    cr, cf = chords[i]
    chord_base_pitch = base_pitch + pitchclass_to_id(cr)
    pitches = [chord_base_pitch + p for p in _CHORDFAMILY_TO_INTERVALS[cf]]
    for p in pitches:
      note = pretty_midi.Note(velocity=80, pitch=p, start=s, end=e)
      cello.notes.append(note)

  midi.instruments.append(cello)

  wav = midi.fluidsynth(fs=fs)
  slen = int(prevlen * fs)
  if prevlen is not None:
    wav = wav[:slen]
    wav = np.pad(wav, [[0, slen - wav.shape[0]]], 'constant')

  return wav.astype(np.float32)


def prev_audio_tf_wrapper(
    prev_fun,
    *args,
    n=None,
    prevlen=10.,
    fs=44100):
  """Wraps calls to audio preview functions for use in tf.summary.audio."""
  if n is not None:
    args = [a[:n] for a in args]

  nsamps = int(prevlen * fs)

  def py_fn_kwarg_closure(*args):
    return prev_fun(*args, prevlen=prevlen, fs=fs)

  def py_fn(*args):
    wav = tf.py_func(
        py_fn_kwarg_closure,
        list(args[0]),
        tf.float32,
        stateful=False)
    wav.set_shape([nsamps])
    return wav

  res = tf.map_fn(py_fn, args, dtype=tf.float32)

  return res


def discrete_to_piano_roll(categorical, dim, dilation=1, colorize=True):
  """Visualizes discrete sequences as a colorful piano roll."""
  # Create piano roll
  if categorical.dtype == tf.int32:
    piano_roll = tf.one_hot(categorical, dim)
  elif categorical.dtype == tf.float32:
    assert int(categorical.get_shape()[-1]) == dim
    piano_roll = categorical
  else:
    raise NotImplementedError()
  piano_roll = tf.stack([piano_roll] * 3, axis=3)

  # Colorize
  if colorize:
    # Create color palette
    hues = np.linspace(0., 1., num=dim, endpoint=False)
    colors_hsv = np.ones([dim, 3], dtype=np.float32)
    colors_hsv[:, 0] = hues
    colors_hsv[:, 1] = 0.85
    colors_hsv[:, 2] = 0.85
    colors_rgb = tf.image.hsv_to_rgb(colors_hsv) * 255.
    colors_rgb = tf.reshape(colors_rgb, [1, 1, dim, 3])

    piano_roll = tf.multiply(piano_roll, colors_rgb)
  else:
    piano_roll *= 255.

  # Rotate and flip for visual ease
  piano_roll = tf.image.rot90(piano_roll)

  # Increase vertical dilation for visual ease
  if dilation > 1:
    old_height = tf.shape(piano_roll)[1]
    old_width = tf.shape(piano_roll)[2]

    piano_roll = tf.image.resize_nearest_neighbor(
        piano_roll, [old_height * dilation, old_width])

  # Cast to tf.uint8
  piano_roll = tf.cast(tf.clip_by_value(piano_roll, 0., 255.), tf.uint8)

  return piano_roll
