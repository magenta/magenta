# Copyright 2024 The Magenta Authors.
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

"""Data parsing functions for tokenized Hooktheory song data."""

import dataclasses
import itertools
from typing import Sequence, Tuple

from magenta.models.realchords import event_codec

MIN_CHORD_IDX = 0
MAX_CHORD_IDX = 28012

REST_NOTE_IDX = 65280
MIN_NOTE_IDX = 65316
MAX_NOTE_IDX = 65443

MAX_STEP = 16383
STEPS_PER_BEAT = 24

# The downsample rate for frame-based representation.
# Originally in Hooktheory dataset, 24 steps = quater notes.
# Here in frame-based representation, a frame = 16th notes.
STEPS_PER_BEAT_FRAME_REP = 4
FRAME_DOWNSAMPLE_RATE = STEPS_PER_BEAT // STEPS_PER_BEAT_FRAME_REP


@dataclasses.dataclass
class Chord:
  idx: int
  start_step: int
  end_step: int


@dataclasses.dataclass
class Note:
  pitch: int
  start_step: int
  end_step: int


@dataclasses.dataclass
class Song:
  chords: Sequence[Chord]
  notes: Sequence[Note]


class SongParseError(Exception):
  pass


def parse_song(tokens: Sequence[int]) -> Song:
  """Parse a song from list of Hooktheory tokens."""
  if len(tokens) % 3 != 0:
    raise ValueError('list of song tokens must have length divisible by 3')
  chords = []
  notes = []
  cur_step = 0
  for idx, start_step, duration in [
      tokens[i : i + 3] for i in range(0, len(tokens), 3)
  ]:
    if start_step + duration > MAX_STEP:
      raise SongParseError(
          f'end step {start_step + duration} greater than maximum step'
          f' {MAX_STEP}'
      )
    if MIN_CHORD_IDX <= idx <= MAX_CHORD_IDX:
      if notes:
        raise SongParseError('unexpected chord after note')
      if start_step < cur_step:
        raise SongParseError(
            f'chord starts at step {start_step} before current step {cur_step}'
        )
      chords.append(
          Chord(
              idx=idx - MIN_CHORD_IDX,
              start_step=start_step,
              end_step=start_step + duration,
          )
      )
    elif MIN_NOTE_IDX <= idx <= MAX_NOTE_IDX:
      if notes and start_step < cur_step:
        raise SongParseError(
            f'note starts at step {start_step} before current step {cur_step}'
        )
      notes.append(
          Note(
              pitch=idx - MIN_NOTE_IDX,
              start_step=start_step,
              end_step=start_step + duration,
          )
      )
    elif idx == REST_NOTE_IDX:
      pass
    else:
      raise SongParseError(f'invalid note or chord index: {idx}')
    cur_step = start_step + duration
  if not chords or not notes:
    raise SongParseError('Song with no chords or notes')
  return Song(chords=chords, notes=notes)


def chords_to_events(chords: Sequence[Chord]) -> Sequence[event_codec.Event]:
  events = [event_codec.Event(type='step', value=0)]
  cur_step = 0
  for chord in chords:
    if chord.start_step > cur_step:
      events.append(event_codec.Event(type='step', value=chord.start_step))
    events.append(event_codec.Event(type='chord', value=chord.idx))
    events.append(event_codec.Event(type='step', value=chord.end_step))
    cur_step = chord.end_step
  return events


def notes_to_events(notes: Sequence[Note]) -> Sequence[event_codec.Event]:
  events = [event_codec.Event(type='step', value=0)]
  cur_step = 0
  for note in notes:
    if note.start_step > cur_step:
      events.append(event_codec.Event(type='step', value=note.start_step))
    events.append(event_codec.Event(type='note', value=note.pitch))
    events.append(event_codec.Event(type='step', value=note.end_step))
    cur_step = note.end_step
  return events


def events_to_chords(events: Sequence[event_codec.Event]) -> Sequence[Chord]:
  """Decode sequence of events to chords."""
  chords = []
  cur_idx = None
  cur_step = 0
  for event in events:
    if event.type == 'step':
      if event.value < cur_step:
        continue
      if cur_idx is not None:
        chords.append(
            Chord(idx=cur_idx, start_step=cur_step, end_step=event.value)
        )
        cur_idx = None
      cur_step = event.value
    elif event.type == 'chord':
      cur_idx = event.value
  return chords


def events_to_notes(events: Sequence[event_codec.Event]) -> Sequence[Note]:
  """Decode sequence of events to notes."""
  notes = []
  cur_pitch = None
  cur_step = 0
  for event in events:
    if event.type == 'step':
      if event.value < cur_step:
        continue
      if cur_pitch is not None:
        notes.append(
            Note(pitch=cur_pitch, start_step=cur_step, end_step=event.value)
        )
        cur_pitch = None
      cur_step = event.value
    elif event.type == 'note':
      cur_pitch = event.value
  return notes


def chords_to_frames(
    chords: Sequence[Chord], downsample_rate: int
) -> Sequence[event_codec.Event]:
  """Convert chords class to list of per-frame chord events."""
  chords_downsampled = [
      Chord(
          idx=c.idx,
          start_step=c.start_step // downsample_rate,
          end_step=c.end_step // downsample_rate,
      )
      for c in chords
  ]
  # +1 to total_duration in case final chord has the same start and end step.
  total_duration = chords[-1].end_step // downsample_rate + 1
  # Initialize frames with rest events.
  frames = [event_codec.Event(type='chord', value=0)] * total_duration
  for chord in chords_downsampled:
    # +1 to chord.idx because we leave 0 for rest.
    frames[chord.start_step : chord.end_step] = [
        event_codec.Event(type='chord', value=chord.idx + 1)
    ] * (chord.end_step - chord.start_step)
    # Add onset frame, but onset event does not have rest, so no +1.
    frames[chord.start_step] = event_codec.Event(
        type='chord_on', value=chord.idx
    )
  return frames


def notes_to_frames(
    notes: Sequence[Note], downsample_rate: int
) -> Sequence[event_codec.Event]:
  """Convert notes class to list of per-frame note events."""
  notes_downsampled = [
      Note(
          pitch=n.pitch,
          start_step=n.start_step // downsample_rate,
          end_step=n.end_step // downsample_rate,
      )
      for n in notes
  ]
  # +1 to total_duration in case final note has the same start and end step.
  total_duration = notes[-1].end_step // downsample_rate + 1
  # Initialize frames with rest events.
  frames = [event_codec.Event(type='note', value=0)] * total_duration
  for note in notes_downsampled:
    # +1 to note.pitch because we leave 0 for rest.
    frames[note.start_step : note.end_step] = [
        event_codec.Event(type='note', value=note.pitch + 1)
    ] * (note.end_step - note.start_step)
    # Add onset frame, but onset event does not have rest, so no +1.
    frames[note.start_step] = event_codec.Event(
        type='note_on', value=note.pitch
    )
  return frames


def frames_to_chords(
    frames: Sequence[int],
    codec: event_codec.Codec,
    upsample_rate: int
) -> Sequence[Chord]:
  """Convert list of per-frame chords to list of chords."""
  all_chords = []
  offset = 0
  current_chord = None
  for token_idx, group in itertools.groupby(frames):
    duration = len(list(group))
    event = codec.decode_event_index(token_idx)
    value = event.value
    if event.type == 'chord_on':
      # In case there are back-to-back chord_on
      for _ in range(duration):
        # Close the current Chord and append to list.
        if current_chord:
          all_chords.append(current_chord)
        current_chord = Chord(
            idx=value,
            start_step=offset * upsample_rate,
            end_step=(offset + 1) * upsample_rate,
        )
        offset += 1
    elif event.type == 'chord':
      if value == 0:  # rest
        if current_chord:
          all_chords.append(current_chord)
        current_chord = None
      else:
        if current_chord:
          if value == current_chord.idx + 1:  # +1 because 0 is used for rest
            # A hold chord
            current_chord.end_step += duration * upsample_rate
          else:
            raise ValueError('Mismatch between chord_on and chord event.')
        else:
          raise ValueError('Chord event without chord_on.')
      offset += duration
    else:
      raise ValueError('All frames must be chords.')

  # Close left-out Chord
  if current_chord:
    all_chords.append(current_chord)
  return all_chords


def frames_to_notes(
    frames: Sequence[int],
    codec: event_codec.Codec,
    upsample_rate: int
) -> Sequence[Note]:
  """Convert list of per-frame notes to list of notes."""
  all_notes = []
  offset = 0
  current_note = None
  for token_idx, group in itertools.groupby(frames):
    duration = len(list(group))
    event = codec.decode_event_index(token_idx)
    value = event.value
    if event.type == 'note_on':
      # In case there are back-to-back note_on.
      for _ in range(duration):
        # Close the current Note and append to list.
        if current_note:
          all_notes.append(current_note)
        current_note = Note(
            pitch=value,
            start_step=offset * upsample_rate,
            end_step=(offset + 1) * upsample_rate,
        )
        offset += 1
    elif event.type == 'note':
      if value == 0:  # rest
        if current_note:
          all_notes.append(current_note)
        current_note = None
      else:
        if current_note:
          if value == current_note.pitch + 1:  # +1 because 0 is used for rest
            # A hold note
            current_note.end_step += duration * upsample_rate
          else:
            raise ValueError('Mismatch between note_on and note event.')
        else:
          raise ValueError('Note event without note_on.')
      offset += duration
    else:
      raise ValueError('All frames must be notes.')

  # Close left-out note
  if current_note:
    all_notes.append(current_note)
  return all_notes


def interleave_chord_note_events(
    chords: Sequence[event_codec.Event],
    notes: Sequence[event_codec.Event],
    model_part: str = 'chord',
) -> Sequence[event_codec.Event]:
  """Interleave chord and note events and merge to a target."""
  # Pad chords and notes to same length.
  max_length = max(len(chords), len(notes))
  chords = [*chords, *[event_codec.Event(type='chord', value=0)] * (
      max_length - len(chords)
  )]
  notes = [*notes, *[event_codec.Event(type='note', value=0)] * (
      max_length - len(notes)
  )]

  # Interleave merge two lists.
  assert len(chords) == len(notes)
  targets = [None] * (len(chords) * 2)
  if model_part == 'chord':
    targets[::2] = chords
    targets[1::2] = notes
  elif model_part == 'note':
    targets[::2] = notes
    targets[1::2] = chords
  else:
    raise ValueError(f'Invalid model_part: {model_part}')
  return targets


def event_list_to_same_length(
    chords: Sequence[event_codec.Event], notes: Sequence[event_codec.Event]
) -> Tuple[Sequence[event_codec.Event], Sequence[event_codec.Event]]:
  """Pad two event lists to have same length."""
  max_length = max(len(chords), len(notes))
  chords_pad = [
      *chords,
      *[event_codec.Event(type='chord', value=0)] * (max_length - len(chords)),
  ]
  notes_pad = [
      *notes,
      *[event_codec.Event(type='note', value=0)] * (max_length - len(notes)),
  ]
  return chords_pad, notes_pad
