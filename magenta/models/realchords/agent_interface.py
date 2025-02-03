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

"""Interface for interacting with generative models.

@Author Alex Scarlatos (scarlatos@google.com)
"""

import json
from typing import List, Optional, Tuple, TypedDict

from absl import logging
import jax
import jax.numpy as jnp
from magenta.models.realchords import data
from magenta.models.realchords import event_codec
from magenta.models.realchords import frame_codec
from magenta.models.realchords import inference_utils
from magenta.models.realchords import non_causal_model_interface
import note_seq
import numpy as np

BASE_MODEL_GIN_FILE_PATH = "gin/base_model.gin"
SMALL_MODEL_GIN_FILE_PATH = "gin/small_model.gin"

_CHORD_MAPPING_FILENAME = (
    "realchords_opensource_assets/chord_mapping.json"
)

CHORD_OCTAVE = 4
BASS_OCTAVE = 3

MODELS = [
    {
        "name": "ReaLchords-S",
        "model_dir": "realchords_opensource_checkpoint/realchords_s",
        "model_step": 20_000,
    },
    {
        "name": "ReaLchords-M",
        "model_dir": "realchords_opensource_checkpoint/realchords_m",
        "model_step": 20_000,
    },
    {
        "name": "Online",
        "model_dir": "realchords_opensource_checkpoint/online",
        "model_step": 50_000,
    },
]

ChordInfo = Tuple[str, List[int], bool]


class NoteInfo(TypedDict):
  pitch: int
  frame: int
  on: bool


class Agent:
  """Interface for interacting with generative chord model."""

  def __init__(self):
    logging.info("Creating model...")
    self.models = {
        model["name"]: inference_utils.MusicTransformerInferenceModel(
            model["model_dir"],
            BASE_MODEL_GIN_FILE_PATH,
            model["model_step"],
            batch_size=1,
            model_parallel_submesh=(1, 1, 1, 1),
            gin_overrides="TASK_FEATURE_LENGTHS={'inputs': 0, 'targets': 512}",
            additional_gin_path=[SMALL_MODEL_GIN_FILE_PATH]
        )
        for model in MODELS
    }
    self.non_causal_model = non_causal_model_interface.NonCausalModel()

    # TODO(alexscarlatos): load the chord mapping from the model dir
    with open(_CHORD_MAPPING_FILENAME, "r") as f:
      self.chord_mapping: List[str] = json.load(f)
    self.codec = frame_codec.get_frame_codec()
    self.chord_rest_token = self.codec.encode_event(
        event_codec.Event(type="chord", value=0)
    )
    self.note_rest_token = self.codec.encode_event(
        event_codec.Event(type="note", value=0)
    )
    self.max_frames = 256
    logging.info("Devices: %s", jax.devices())

  def get_models(self) -> List[str]:
    return list(self.models.keys())

  def decode_chord_token(self, chord_token: int) -> ChordInfo:
    """Extract the underlying information from a given chord token.

    Args:
      chord_token: the chord token to decode

    Returns:
      chord_symbol: string symbolic representation of chord
      chord_pitches: list of pitches for chord
      is_onset: if the chord is a hit or a hold
    """
    if chord_token == self.chord_rest_token:
      return "", [], True  # No pitches for rest
    # Get pitches for chord token
    new_chord_event = self.codec.decode_event_index(chord_token)
    is_onset = new_chord_event.type == "chord_on"
    new_chord_id = (
        new_chord_event.value - 1
        if new_chord_event.type == "chord"
        else new_chord_event.value
    )
    chord_symbol = self.chord_mapping[new_chord_id]
    try:
      chord_pitches = note_seq.chord_symbol_pitches(chord_symbol)
      bass_pitch = note_seq.chord_symbol_bass(chord_symbol)
      pitches = [
          *[CHORD_OCTAVE * 12 + pitch for pitch in chord_pitches],
          BASS_OCTAVE * 12 + bass_pitch,
      ]
    except (note_seq.ChordSymbolError, TypeError) as e:
      logging.warning(
          "Failed to convert chord symbol %s to pitches: %s", chord_symbol, e
      )
      chord_symbol = ""
      pitches = []
    return chord_symbol, pitches, is_onset

  def melody_to_frame_tokens(
      self, melody_data: List[NoteInfo], end_frame: int = 0
  ) -> List[int]:
    """Given melody, convert to homophony and return frame-based tokens.

    Args:
      melody_data: list of note on/off events from frontend
      end_frame: frame to pad till or cut off at (if > 0)

    Returns:
      list of note tokens (rest/onset/hold), one for each frame
    """
    cur_pitch = None
    cur_start_step = None
    notes = []
    for event in melody_data:
      if event["on"]:
        # End currently held note if playing new note
        if cur_pitch is not None and cur_pitch != event["pitch"]:
          notes.append(
              data.Note(
                  pitch=cur_pitch,
                  start_step=cur_start_step,
                  end_step=event["frame"],
              )
          )
        # Set currently held note
        cur_pitch = event["pitch"]
        cur_start_step = event["frame"]
      elif cur_pitch == event["pitch"]:
        # End currently held note on release
        notes.append(
            data.Note(
                pitch=cur_pitch,
                start_step=cur_start_step,
                end_step=event["frame"],
            )
        )
        # Unset currently held note
        cur_pitch = None
        cur_start_step = None

    # If note was being held at end of sequence then end at last frame
    if end_frame and cur_pitch is not None:
      notes.append(
          data.Note(
              pitch=cur_pitch,
              start_step=cur_start_step,
              end_step=end_frame,
          )
      )

    # Convert to frames and tokenize
    frames = data.notes_to_frames(notes, downsample_rate=1)
    tokens = [self.codec.encode_event(frame) for frame in frames]
    if end_frame:
      # Remove padding past last frame
      tokens = tokens[:end_frame]
      # Pad up to last frame with rests (needed if no active note)
      if len(tokens) < end_frame:
        tokens.extend([self.note_rest_token] * (end_frame - len(tokens)))
    return tokens

  def generate_live(
      self,
      model_name: str,
      notes: List[NoteInfo],
      chord_tokens: List[int],
      frame: int,
      lookahead: int,
      commitahead: int,
      temperature: float,
      silence_till: int,
      intro_set: bool,
  ) -> Tuple[List[ChordInfo], List[int], Optional[List[int]]]:
    """Generate chords for a given frame.

    - Condition on melody and previously generated chord tokens.
    - Generate lookahead frames into the future.
    - Force chords to not change during commitahead from previous generation.
    - Generate predicted intro chords before silence cutoff.

    Args:
      model_name: model to use, from MODELS keys
      notes: list of all note events in session
      chord_tokens: list of all chord tokens in session
      frame: frame to start generating at
      lookahead: how many frames into the future to generate
      commitahead: how many frames to leave chords unchanged since last call
      temperature: model sampling temperature
      silence_till: frames before generating with online model
      intro_set: if intro chords have been filled in by offline model

    Returns:
      new_chords: list of ChordInfo, one for each frame in lookahead
      new_chord_tokens: list of chord tokens, one for each frame in lookahead
      intro_chord_tokens: list of chord tokens to fill in session beginning,
        only returned when first generated right before silence cutoff frame
    """
    # Wait until at most 4 frames before start to have enough context
    gen_start_frame = silence_till - min(lookahead, 4)
    if frame < gen_start_frame or frame == 0:
      logging.info(
          "Waiting for more user input before generating; frame: %s, lookahead:"
          " %s, silence_till: %s",
          frame,
          lookahead,
          silence_till,
      )
      return [], [], None

    logging.info("Generate chords for frame %s", frame)

    model = self.models[model_name]

    # Convert notes to frame token format
    note_token_hist = self.melody_to_frame_tokens(notes, frame)
    # logging.info("Note token hist: %s", note_token_hist)

    # Use non-causal model to generate likely chords for introduction section
    intro_chord_tokens = None
    if not intro_set:
      intro_chord_tokens = self.non_causal_model.get_output_tokens(
          note_token_hist, 0.0
      )
      chord_tokens[:frame] = intro_chord_tokens
      logging.info("Intro chord tokens: %s", intro_chord_tokens)

    # Fill any gaps in chord tokens with rests
    for i in range(len(chord_tokens)):
      if chord_tokens[i] == -1:
        chord_tokens[i] = self.chord_rest_token
    if len(chord_tokens) < frame:
      chord_tokens.extend([self.chord_rest_token] * (frame - len(chord_tokens)))

    # Create initial prompt (up to target frame)
    chord_token_hist = chord_tokens[:frame]
    # Trim beginning of context to avoid surpassing max length
    max_context_len = self.max_frames - lookahead
    note_token_hist = note_token_hist[-max_context_len:]
    chord_token_hist = chord_token_hist[-max_context_len:]
    prompt = np.full(
        (1, 2 * len(chord_token_hist)), self.chord_rest_token, dtype=np.int32
    )
    prompt[0, 0::2] = chord_token_hist
    prompt[0, 1::2] = note_token_hist
    start_idx = prompt.shape[1]
    logging.info("Prompt: %s", prompt)

    # First predict up to commit point to get predicted user notes
    commit_frames = len(chord_tokens) - lookahead + commitahead - frame
    commit_frames = max(commit_frames, 0)  # If frame is past commit point
    if commit_frames >= lookahead:
      # Can just look up new tokens in cache if lookahead was decreased by user
      # Otherwise will try to generate <= 0 tokens and will break model api
      new_chord_tokens = np.array(
          chord_tokens[frame : frame + min(commit_frames, lookahead)]
      )
    else:
      if commit_frames > 0:
        new_tokens = model(
            decoder_prompt=jnp.array(prompt),
            decode_length=start_idx + commit_frames * 2,
            seed=np.random.randint(1234),
            # Seems like type checker has bug handling kwargs type
            temperature=temperature,  # type: ignore
        )
        new_tokens = np.array(new_tokens)
        # Replace predicted chords with committed chords
        new_tokens[0, 0::2] = chord_tokens[frame : frame + commit_frames]
        # Update prompt to commit point
        prompt = np.concatenate([prompt, new_tokens], axis=1)

      # Then predict up to lookahead to get new chord predictions
      new_tokens = model(
          decoder_prompt=jnp.array(prompt),
          decode_length=start_idx + lookahead * 2 - 1,
          seed=np.random.randint(1234),
          # Seems like type checker has bug handling kwargs type
          temperature=temperature,  # type: ignore
      )
      new_tokens = np.array(new_tokens)
      all_tokens = np.concatenate([prompt, new_tokens], axis=1)
      new_chord_tokens = all_tokens[0, start_idx::2]

    # Decode and return new chord tokens
    new_chords = [
        self.decode_chord_token(chord_tok) for chord_tok in new_chord_tokens
    ]
    logging.info("Chords: %s, cur_frame: %d", new_chords, frame)
    return new_chords, new_chord_tokens.tolist(), intro_chord_tokens
