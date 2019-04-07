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


def notes_to_prev_audio(
    midi_pitches,
    start_times,
    end_times,
    prevlen=10.,
    fs=44100):
  """Creates a preview waveform from piano notes."""
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


def prev_audio_tf_wrapper(
    prev_fun,
    *args,
    n=None,
    prevlen=10.,
    fs=44100):
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
