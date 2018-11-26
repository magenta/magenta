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

"""Transcribe a recording of piano audio."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import librosa
import tensorflow as tf

from magenta.common import tf_utils
from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription import data
from magenta.models.onsets_frames_transcription import model
from magenta.music import audio_io
from magenta.music import midi_io
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'acoustic_run_dir', None,
    'Path to look for acoustic checkpoints. Should contain subdir `train`.')
tf.app.flags.DEFINE_string(
    'acoustic_checkpoint_filename', None,
    'Filename of the checkpoint to use. If not specified, will use the latest '
    'checkpoint')
tf.app.flags.DEFINE_string(
    'hparams',
    'onset_mode=length_ms,onset_length=32',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_float(
    'frame_threshold', 0.5,
    'Threshold to use when sampling from the acoustic model.')
tf.app.flags.DEFINE_float(
    'onset_threshold', 0.5,
    'Threshold to use when sampling from the acoustic model.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')


def create_example(filename, hparams):
  """Processes an audio file into an Example proto."""
  wav_data = librosa.core.load(filename, sr=hparams.sample_rate)[0]
  if hparams.normalize_audio:
    audio_io.normalize_wav_data(wav_data, hparams.sample_rate)
  wav_data = audio_io.samples_to_wav_data(wav_data, hparams.sample_rate)

  example = tf.train.Example(features=tf.train.Features(feature={
      'id':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[filename.encode('utf-8')]
          )),
      'sequence':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[music_pb2.NoteSequence().SerializeToString()]
          )),
      'audio':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[wav_data]
          )),
      'velocity_range':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[music_pb2.VelocityRange().SerializeToString()]
          )),
  }))

  return example.SerializeToString()


TranscriptionSession = collections.namedtuple(
    'TranscriptionSession',
    ('session', 'examples', 'iterator', 'onset_probs_flat', 'frame_probs_flat',
     'velocity_values_flat', 'hparams'))


def initialize_session(acoustic_checkpoint, hparams):
  """Initializes a transcription session."""
  with tf.Graph().as_default():
    examples = tf.placeholder(tf.string, [None])

    batch, iterator = data.provide_batch(
        batch_size=1,
        examples=examples,
        hparams=hparams,
        is_training=False,
        truncated_length=0)

    model.get_model(batch, hparams, is_training=False)

    session = tf.Session()
    saver = tf.train.Saver()
    saver.restore(session, acoustic_checkpoint)

    onset_probs_flat = tf.get_default_graph().get_tensor_by_name(
        'onsets/onset_probs_flat:0')
    frame_probs_flat = tf.get_default_graph().get_tensor_by_name(
        'frame_probs_flat:0')
    velocity_values_flat = tf.get_default_graph().get_tensor_by_name(
        'velocity/velocity_values_flat:0')

    return TranscriptionSession(
        session=session,
        examples=examples,
        iterator=iterator,
        onset_probs_flat=onset_probs_flat,
        frame_probs_flat=frame_probs_flat,
        velocity_values_flat=velocity_values_flat,
        hparams=hparams)


def transcribe_audio(transcription_session, filename, frame_threshold,
                     onset_threshold):
  """Transcribes an audio file."""
  tf.logging.info('Processing file...')
  transcription_session.session.run(
      transcription_session.iterator.initializer,
      {transcription_session.examples: [
          create_example(filename, transcription_session.hparams)]})
  tf.logging.info('Running inference...')
  frame_logits, onset_logits, velocity_values = (
      transcription_session.session.run([
          transcription_session.frame_probs_flat,
          transcription_session.onset_probs_flat,
          transcription_session.velocity_values_flat]))

  frame_predictions = frame_logits > frame_threshold

  onset_predictions = onset_logits > onset_threshold

  sequence_prediction = sequences_lib.pianoroll_to_note_sequence(
      frame_predictions,
      frames_per_second=data.hparams_frames_per_second(
          transcription_session.hparams),
      min_duration_ms=0,
      onset_predictions=onset_predictions,
      velocity_values=velocity_values)

  for note in sequence_prediction.notes:
    note.pitch += constants.MIN_MIDI_PITCH

  return sequence_prediction


def main(argv):
  tf.logging.set_verbosity(FLAGS.log)

  if FLAGS.acoustic_checkpoint_filename:
    acoustic_checkpoint = os.path.join(
        os.path.expanduser(FLAGS.acoustic_run_dir), 'train',
        FLAGS.acoustic_checkpoint_filename)
  else:
    acoustic_checkpoint = tf.train.latest_checkpoint(
        os.path.join(os.path.expanduser(FLAGS.acoustic_run_dir), 'train'))

  hparams = tf_utils.merge_hparams(
      constants.DEFAULT_HPARAMS, model.get_default_hparams())
  hparams.parse(FLAGS.hparams)

  transcription_session = initialize_session(acoustic_checkpoint, hparams)

  for filename in argv[1:]:
    tf.logging.info('Starting transcription for %s...', filename)

    sequence_prediction = transcribe_audio(
        transcription_session, filename, FLAGS.frame_threshold,
        FLAGS.onset_threshold)

    midi_filename = filename + '.midi'
    midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)

    tf.logging.info('Transcription written to %s.', midi_filename)


def console_entry_point():
  tf.app.run(main)

if __name__ == '__main__':
  console_entry_point()
