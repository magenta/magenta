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

"""Transcribe a recording of piano audio."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from magenta.models.onsets_frames_transcription import audio_label_data_utils
from magenta.models.onsets_frames_transcription import configs
from magenta.models.onsets_frames_transcription import data
from magenta.models.onsets_frames_transcription import infer_util
from magenta.models.onsets_frames_transcription import train_util
from note_seq import midi_io
from note_seq.protobuf import music_pb2
import six
import tensorflow.compat.v1 as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', 'onsets_frames',
                           'Name of the config to use.')
tf.app.flags.DEFINE_string('model_dir', None,
                           'Path to look for acoustic checkpoints.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'Filename of the checkpoint to use. If not specified, will use the latest '
    'checkpoint')
tf.app.flags.DEFINE_string(
    'hparams',
    '',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_boolean(
    'load_audio_with_librosa', False,
    'Whether to use librosa for sampling audio (required for 24-bit audio)')
tf.app.flags.DEFINE_string(
    'transcribed_file_suffix', '',
    'Optional suffix to add to transcribed files.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')


def create_example(filename, sample_rate, load_audio_with_librosa):
  """Processes an audio file into an Example proto."""
  wav_data = tf.gfile.Open(filename, 'rb').read()
  example_list = list(
      audio_label_data_utils.process_record(
          wav_data=wav_data,
          sample_rate=sample_rate,
          ns=music_pb2.NoteSequence(),
          # decode to handle filenames with extended characters.
          example_id=six.ensure_text(filename, 'utf-8'),
          min_length=0,
          max_length=-1,
          allow_empty_notesequence=True,
          load_audio_with_librosa=load_audio_with_librosa))
  assert len(example_list) == 1
  return example_list[0].SerializeToString()


def run(argv, config_map, data_fn):
  """Create transcriptions."""
  tf.logging.set_verbosity(FLAGS.log)

  config = config_map[FLAGS.config]
  hparams = config.hparams
  hparams.parse(FLAGS.hparams)
  hparams.batch_size = 1
  hparams.truncated_length_secs = 0

  with tf.Graph().as_default():
    examples = tf.placeholder(tf.string, [None])

    dataset = data_fn(
        examples=examples,
        preprocess_examples=True,
        params=hparams,
        is_training=False,
        shuffle_examples=False,
        skip_n_initial_records=0)

    estimator = train_util.create_estimator(config.model_fn,
                                            os.path.expanduser(FLAGS.model_dir),
                                            hparams)

    iterator = tf.data.make_initializable_iterator(dataset)
    next_record = iterator.get_next()

    with tf.Session() as sess:
      sess.run([
          tf.initializers.global_variables(),
          tf.initializers.local_variables()
      ])

      for filename in argv[1:]:
        tf.logging.info('Starting transcription for %s...', filename)

        # The reason we bounce between two Dataset objects is so we can use
        # the data processing functionality in data.py without having to
        # construct all the Example protos in memory ahead of time or create
        # a temporary tfrecord file.
        tf.logging.info('Processing file...')
        sess.run(iterator.initializer,
                 {examples: [
                     create_example(filename, hparams.sample_rate,
                                    FLAGS.load_audio_with_librosa)]})

        def transcription_data(params):
          del params
          return tf.data.Dataset.from_tensors(sess.run(next_record))
        input_fn = infer_util.labels_to_features_wrapper(transcription_data)

        tf.logging.info('Running inference...')
        checkpoint_path = None
        if FLAGS.checkpoint_path:
          checkpoint_path = os.path.expanduser(FLAGS.checkpoint_path)
        prediction_list = list(
            estimator.predict(
                input_fn,
                checkpoint_path=checkpoint_path,
                yield_single_examples=False))
        assert len(prediction_list) == 1

        sequence_prediction = music_pb2.NoteSequence.FromString(
            prediction_list[0]['sequence_predictions'][0])

        midi_filename = filename + FLAGS.transcribed_file_suffix + '.midi'
        midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)

        tf.logging.info('Transcription written to %s.', midi_filename)


def main(argv):
  run(argv, config_map=configs.CONFIG_MAP, data_fn=data.provide_batch)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)

if __name__ == '__main__':
  console_entry_point()
