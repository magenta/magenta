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
"""Inference for onset conditioned model.

A histogram summary will be written for every example processed, and the
resulting MIDI and pianoroll images will also be written for every example.
The final summary value is the mean score for all examples.
"""

import collections
import functools
import os
import time
import imageio
from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription import data
from magenta.models.onsets_frames_transcription import infer_util
from magenta.models.onsets_frames_transcription import train_util
from note_seq import midi_io
from note_seq import sequences_lib
from note_seq.protobuf import music_pb2
import numpy as np
import six
import tensorflow.compat.v1 as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('master', '',
                           'Name of the TensorFlow runtime to use.')
tf.app.flags.DEFINE_string('config', 'onsets_frames',
                           'Name of the config to use.')
tf.app.flags.DEFINE_string('model_dir', None, 'Path to look for checkpoints.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'Filename of the checkpoint to use. If not specified, will use the latest '
    'checkpoint')
tf.app.flags.DEFINE_string('examples_path', None,
                           'Path to test examples TFRecord.')
tf.app.flags.DEFINE_string(
    'output_dir', '~/tmp/onsets_frames/infer',
    'Path to store output midi files and summary events.')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_boolean(
    'shuffle_examples', False, 'Whether to shuffle examples.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')
tf.app.flags.DEFINE_boolean('preprocess_examples', False,
                            'Whether or not to run preprocessing on examples.')


def model_inference(model_fn,
                    model_dir,
                    checkpoint_path,
                    data_fn,
                    hparams,
                    examples_path,
                    output_dir,
                    summary_writer,
                    master,
                    preprocess_examples,
                    shuffle_examples):
  """Runs inference for the given examples."""
  tf.logging.info('model_dir=%s', model_dir)
  tf.logging.info('checkpoint_path=%s', checkpoint_path)
  tf.logging.info('examples_path=%s', examples_path)
  tf.logging.info('output_dir=%s', output_dir)

  estimator = train_util.create_estimator(
      model_fn, model_dir, hparams, master=master)

  transcription_data = functools.partial(
      data_fn, examples=examples_path, preprocess_examples=preprocess_examples,
      is_training=False, shuffle_examples=shuffle_examples,
      skip_n_initial_records=0)

  input_fn = infer_util.labels_to_features_wrapper(transcription_data)

  start_time = time.time()
  infer_times = []
  num_frames = []

  file_num = 0

  all_metrics = collections.defaultdict(list)

  for predictions in estimator.predict(
      input_fn, checkpoint_path=checkpoint_path, yield_single_examples=False):

    # Remove batch dimension for convenience.
    for k in predictions.keys():
      if predictions[k].shape[0] != 1:
        raise ValueError(
            'All predictions must have batch size 1, but shape of '
            '{} was: {}'.format(k, + predictions[k].shape[0]))
      predictions[k] = predictions[k][0]

    end_time = time.time()
    infer_time = end_time - start_time
    infer_times.append(infer_time)
    num_frames.append(predictions['frame_predictions'].shape[0])
    tf.logging.info(
        'Infer time %f, frames %d, frames/sec %f, running average %f',
        infer_time, num_frames[-1], num_frames[-1] / infer_time,
        np.sum(num_frames) / np.sum(infer_times))

    tf.logging.info('Scoring sequence %s', predictions['sequence_ids'])

    sequence_prediction = music_pb2.NoteSequence.FromString(
        predictions['sequence_predictions'])
    sequence_label = music_pb2.NoteSequence.FromString(
        predictions['sequence_labels'])

    # Make filenames UNIX-friendly.
    filename_chars = six.ensure_text(predictions['sequence_ids'], 'utf-8')
    filename_chars = [c if c.isalnum() else '_' for c in filename_chars]
    filename_safe = ''.join(filename_chars).rstrip()
    filename_safe = '{:04d}_{}'.format(file_num, filename_safe[:200])
    file_num += 1
    output_file = os.path.join(output_dir, filename_safe + '.mid')
    tf.logging.info('Writing inferred midi file to %s', output_file)
    midi_io.sequence_proto_to_midi_file(sequence_prediction, output_file)

    label_output_file = os.path.join(output_dir, filename_safe + '_label.mid')
    tf.logging.info('Writing label midi file to %s', label_output_file)
    midi_io.sequence_proto_to_midi_file(sequence_label, label_output_file)

    # Also write a pianoroll showing acoustic model output vs labels.
    pianoroll_output_file = os.path.join(
        output_dir, filename_safe + '_pianoroll.png')
    tf.logging.info('Writing acoustic logit/label file to %s',
                    pianoroll_output_file)
    # Calculate frames based on the sequence. Includes any postprocessing done
    # to turn raw onsets/frames predictions into the final sequence.
    # TODO(fjord): This work is duplicated in metrics.py.
    sequence_frame_predictions = sequences_lib.sequence_to_pianoroll(
        sequence_prediction,
        frames_per_second=data.hparams_frames_per_second(hparams),
        min_pitch=constants.MIN_MIDI_PITCH,
        max_pitch=constants.MAX_MIDI_PITCH).active
    with tf.gfile.GFile(pianoroll_output_file, mode='w') as f:
      imageio.imwrite(
          f,
          infer_util.posterior_pianoroll_image(
              predictions['onset_probs'],
              predictions['onset_labels'],
              predictions['frame_probs'],
              predictions['frame_labels'],
              sequence_frame_predictions),
          format='png')

    # Update histogram and current scalar for metrics.
    with tf.Graph().as_default(), tf.Session().as_default():
      for k, v in predictions.items():
        if not k.startswith('metrics/'):
          continue
        all_metrics[k].extend(v)
        histogram_name = k + '_histogram'
        metric_summary = tf.summary.histogram(histogram_name, all_metrics[k])
        summary_writer.add_summary(metric_summary.eval(), global_step=file_num)
        scalar_name = k
        metric_summary = tf.summary.scalar(scalar_name, np.mean(all_metrics[k]))
        summary_writer.add_summary(metric_summary.eval(), global_step=file_num)
      summary_writer.flush()

    start_time = time.time()

  # Write final mean values for all metrics.
  with tf.Graph().as_default(), tf.Session().as_default():
    for k, v in all_metrics.items():
      final_scalar_name = 'final/' + k
      metric_summary = tf.summary.scalar(
          final_scalar_name, np.mean(all_metrics[k]))
      summary_writer.add_summary(metric_summary.eval())
    summary_writer.flush()


def run(config_map, data_fn):
  """Run the infer script."""
  output_dir = os.path.expanduser(FLAGS.output_dir)

  config = config_map[FLAGS.config]
  hparams = config.hparams
  hparams.parse(FLAGS.hparams)

  # Batch size should always be 1 for inference.
  hparams.batch_size = 1

  tf.logging.info(hparams)

  tf.gfile.MakeDirs(output_dir)

  summary_writer = tf.summary.FileWriter(logdir=output_dir)

  with tf.Session():
    run_config = '\n\n'.join([
        'model_dir: ' + FLAGS.model_dir,
        'checkpoint_path: ' + str(FLAGS.checkpoint_path),
        'examples_path: ' + FLAGS.examples_path,
        str(hparams),
    ])
    run_config_summary = tf.summary.text(
        'run_config',
        tf.constant(run_config, name='run_config'),
        collections=[])
    summary_writer.add_summary(run_config_summary.eval())

  model_inference(
      model_fn=config.model_fn,
      model_dir=FLAGS.model_dir,
      checkpoint_path=FLAGS.checkpoint_path,
      data_fn=data_fn,
      hparams=hparams,
      examples_path=FLAGS.examples_path,
      output_dir=output_dir,
      summary_writer=summary_writer,
      preprocess_examples=FLAGS.preprocess_examples,
      master=FLAGS.master,
      shuffle_examples=FLAGS.shuffle_examples)
