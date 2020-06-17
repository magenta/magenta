# Copyright 2020 The Magenta Authors.
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

# Lint as: python3
"""Helper functions for generating sounds.
"""

from magenta.models.gansynth.lib import util
import note_seq
import numpy as np
import scipy.io.wavfile as wavfile

MAX_NOTE_LENGTH = 3.0
MAX_VELOCITY = 127.0


def slerp(p0, p1, t):
  """Spherical linear interpolation."""
  omega = np.arccos(np.dot(
      np.squeeze(p0/np.linalg.norm(p0)), np.squeeze(p1/np.linalg.norm(p1))))
  so = np.sin(omega)
  return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1


def load_midi(midi_path, min_pitch=36, max_pitch=84):
  """Load midi as a notesequence."""
  midi_path = util.expand_path(midi_path)
  ns = note_seq.midi_file_to_sequence_proto(midi_path)
  pitches = np.array([n.pitch for n in ns.notes])
  velocities = np.array([n.velocity for n in ns.notes])
  start_times = np.array([n.start_time for n in ns.notes])
  end_times = np.array([n.end_time for n in ns.notes])
  valid = np.logical_and(pitches >= min_pitch, pitches <= max_pitch)
  notes = {'pitches': pitches[valid],
           'velocities': velocities[valid],
           'start_times': start_times[valid],
           'end_times': end_times[valid]}
  return ns, notes


def get_random_instruments(model, total_time, secs_per_instrument=2.0):
  """Get random latent vectors evenly spaced in time."""
  n_instruments = int(total_time / secs_per_instrument)
  z_instruments = model.generate_z(n_instruments)
  t_instruments = np.linspace(-.0001, total_time, n_instruments)
  return z_instruments, t_instruments


def get_z_notes(start_times, z_instruments, t_instruments):
  """Get interpolated latent vectors for each note."""
  z_notes = []
  for t in start_times:
    idx = np.searchsorted(t_instruments, t, side='left') - 1

    # Handles out of bounds errors.
    if idx.item() == t_instruments.size - 1:
      idx -= 1

    t_left = t_instruments[idx]
    t_right = t_instruments[idx + 1]
    interp = (t - t_left) / (t_right - t_left)
    z_notes.append(slerp(z_instruments[idx], z_instruments[idx + 1], interp))
  z_notes = np.vstack(z_notes)
  return z_notes


def get_envelope(t_note_length, t_attack=0.010, t_release=0.3, sr=16000):
  """Create an attack sustain release amplitude envelope."""
  t_note_length = min(t_note_length, MAX_NOTE_LENGTH)
  i_attack = int(sr * t_attack)
  i_sustain = int(sr * t_note_length)
  i_release = int(sr * t_release)
  i_tot = i_sustain + i_release  # attack envelope doesn't add to sound length
  envelope = np.ones(i_tot)
  # Linear attack
  envelope[:i_attack] = np.linspace(0.0, 1.0, i_attack)
  # Linear release
  envelope[i_sustain:i_tot] = np.linspace(1.0, 0.0, i_release)
  return envelope


def combine_notes(audio_notes, start_times, end_times, velocities, sr=16000):
  """Combine audio from multiple notes into a single audio clip.

  Args:
    audio_notes: Array of audio [n_notes, audio_samples].
    start_times: Array of note starts in seconds [n_notes].
    end_times: Array of note ends in seconds [n_notes].
    velocities: Array of velocity values [n_notes].
    sr: Integer, sample rate.

  Returns:
    audio_clip: Array of combined audio clip [audio_samples]
  """
  n_notes = len(audio_notes)
  clip_length = end_times.max() + MAX_NOTE_LENGTH
  audio_clip = np.zeros(int(clip_length) * sr)

  for t_start, t_end, vel, i in zip(
      start_times, end_times, velocities, range(n_notes)):
    # Generate an amplitude envelope
    t_note_length = t_end - t_start
    envelope = get_envelope(t_note_length)
    length = len(envelope)
    audio_note = audio_notes[i, :length] * envelope
    # Normalize
    audio_note /= audio_note.max()
    audio_note *= (vel / MAX_VELOCITY)
    # Add to clip buffer
    clip_start = int(t_start * sr)
    clip_end = clip_start + length
    audio_clip[clip_start:clip_end] += audio_note

  # Normalize
  audio_clip /= audio_clip.max()
  audio_clip /= 2.0
  return audio_clip


def save_wav(audio, fname, sr=16000):
  wavfile.write(fname, sr, audio.astype('float32'))
  print('Saved to {}'.format(fname))
