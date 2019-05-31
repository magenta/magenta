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

"""Inference for onset conditioned model.

Can be used to get a final inference score for a trained model with
--eval_loop=False. In this case, a summary will be written for every example
processed, and the resulting MIDI and pianoroll images will also be written
for every example. The final summary value is the mean score for all examples.

Or, with --eval_loop=True, the script will watch for new checkpoints and re-run
the same scoring code over every example, but only the final mean scores will be
written for every checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription import data
from magenta.models.onsets_frames_transcription import infer_util
from magenta.models.onsets_frames_transcription import train_util
from magenta.music import midi_io
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2

import numpy as np
import scipy
import tensorflow as tf


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
tf.app.flags.DEFINE_integer(
    'max_seconds_per_sequence', None,
    'If set, will truncate sequences to be at most this many seconds long.')
tf.app.flags.DEFINE_boolean(
    'require_onset', True,
    'If set, require an onset prediction for a new note to start.')
tf.app.flags.DEFINE_boolean(
    'use_offset', False,
    'If set, use the offset predictions to force notes to end.')
tf.app.flags.DEFINE_boolean(
    'eval_loop', False,
    'If set, will re-run inference every time a new checkpoint is written. '
    'And will only output the final results.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')
tf.app.flags.DEFINE_boolean('preprocess_examples', False,
                            'Whether or not to run preprocessing on examples.')


def model_inference(model_fn,
                    model_dir,
                    checkpoint_path,
                    hparams,
                    examples_path,
                    output_dir,
                    summary_writer,
                    master,
                    preprocess_examples,
                    write_summary_every_step=True):
  """Runs inference for the given examples."""
  tf.logging.info('model_dir=%s', model_dir)
  tf.logging.info('checkpoint_path=%s', checkpoint_path)
  tf.logging.info('examples_path=%s', examples_path)
  tf.logging.info('output_dir=%s', output_dir)

  estimator = train_util.create_estimator(
      model_fn, model_dir, hparams, master=master)

  with tf.Graph().as_default():
    num_dims = constants.MIDI_PITCHES

    dataset = data.provide_batch(
        examples=examples_path,
        preprocess_examples=preprocess_examples,
        params=hparams,
        is_training=False,
        shuffle_examples=False,
        skip_n_initial_records=0)

    # Define some metrics.
    (metrics_to_updates, metric_note_precision, metric_note_recall,
     metric_note_f1, metric_note_precision_with_offsets,
     metric_note_recall_with_offsets, metric_note_f1_with_offsets,
     metric_note_precision_with_offsets_velocity,
     metric_note_recall_with_offsets_velocity,
     metric_note_f1_with_offsets_velocity, metric_frame_labels,
     metric_frame_predictions) = infer_util.define_metrics(num_dims)

    summary_op = tf.summary.merge_all()

    if write_summary_every_step:
      global_step = tf.train.get_or_create_global_step()
      global_step_increment = global_step.assign_add(1)
    else:
      global_step = tf.constant(
          estimator.get_variable_value(tf.GraphKeys.GLOBAL_STEP))
      global_step_increment = global_step

    iterator = dataset.make_initializable_iterator()
    next_record = iterator.get_next()
    with tf.Session() as sess:
      sess.run([
          tf.initializers.global_variables(),
          tf.initializers.local_variables()
      ])

      infer_times = []
      num_frames = []

      sess.run(iterator.initializer)
      while True:
        try:
          record = sess.run(next_record)
        except tf.errors.OutOfRangeError:
          break

        def input_fn(params):
          del params
          return tf.data.Dataset.from_tensors(record)

        start_time = time.time()

        # TODO(fjord): This is a hack that allows us to keep using our existing
        # infer/scoring code with a tf.Estimator model. Ideally, we should
        # move things around so that we can use estimator.evaluate, which will
        # also be more efficient because it won't have to restore the checkpoint
        # for every example.
        prediction_list = list(
            estimator.predict(
                input_fn,
                checkpoint_path=checkpoint_path,
                yield_single_examples=False))
        assert len(prediction_list) == 1

        input_features = record[0]
        input_labels = record[1]

        filename = input_features.sequence_id[0]
        note_sequence = music_pb2.NoteSequence.FromString(
            input_labels.note_sequence[0])
        labels = input_labels.labels[0]
        frame_probs = prediction_list[0]['frame_probs'][0]
        frame_predictions = prediction_list[0]['frame_predictions'][0]
        onset_predictions = prediction_list[0]['onset_predictions'][0]
        velocity_values = prediction_list[0]['velocity_values'][0]
        offset_predictions = prediction_list[0]['offset_predictions'][0]

        if not FLAGS.require_onset:
          onset_predictions = None

        if not FLAGS.use_offset:
          offset_predictions = None

        sequence_prediction = sequences_lib.pianoroll_to_note_sequence(
            frame_predictions,
            frames_per_second=data.hparams_frames_per_second(hparams),
            min_duration_ms=0,
            min_midi_pitch=constants.MIN_MIDI_PITCH,
            onset_predictions=onset_predictions,
            offset_predictions=offset_predictions,
            velocity_values=velocity_values)

        end_time = time.time()
        infer_time = end_time - start_time
        infer_times.append(infer_time)
        num_frames.append(frame_predictions.shape[0])
        tf.logging.info(
            'Infer time %f, frames %d, frames/sec %f, running average %f',
            infer_time, frame_predictions.shape[0],
            frame_predictions.shape[0] / infer_time,
            np.sum(num_frames) / np.sum(infer_times))

        tf.logging.info('Scoring sequence %s', filename)

        def shift_notesequence(ns_time):
          return ns_time + hparams.backward_shift_amount_ms / 1000.

        sequence_label = sequences_lib.adjust_notesequence_times(
            note_sequence, shift_notesequence)[0]
        infer_util.score_sequence(
            sess,
            global_step_increment,
            metrics_to_updates,
            metric_note_precision,
            metric_note_recall,
            metric_note_f1,
            metric_note_precision_with_offsets,
            metric_note_recall_with_offsets,
            metric_note_f1_with_offsets,
            metric_note_precision_with_offsets_velocity,
            metric_note_recall_with_offsets_velocity,
            metric_note_f1_with_offsets_velocity,
            metric_frame_labels,
            metric_frame_predictions,
            frame_labels=labels,
            sequence_prediction=sequence_prediction,
            frames_per_second=data.hparams_frames_per_second(hparams),
            sequence_label=sequence_label,
            sequence_id=filename)

        if write_summary_every_step:
          # Make filenames UNIX-friendly.
          chars = filename.decode('utf-8')
          chars = [c if c.isalnum() else '_' for c in chars]
          filename_safe = ''.join(chars).rstrip()
          output_file = os.path.join(output_dir, filename_safe + '.mid')
          tf.logging.info('Writing inferred midi file to %s', output_file)
          midi_io.sequence_proto_to_midi_file(sequence_prediction, output_file)

          label_output_file = os.path.join(output_dir,
                                           filename_safe + '_label.mid')
          tf.logging.info('Writing label midi file to %s', label_output_file)
          midi_io.sequence_proto_to_midi_file(sequence_label, label_output_file)

          # Also write a pianoroll showing acoustic model output vs labels.
          pianoroll_output_file = os.path.join(output_dir,
                                               filename_safe + '_pianoroll.png')
          tf.logging.info('Writing acoustic logit/label file to %s',
                          pianoroll_output_file)
          with tf.gfile.GFile(pianoroll_output_file, mode='w') as f:
            scipy.misc.imsave(
                f,
                infer_util.posterior_pianoroll_image(
                    frame_probs,
                    sequence_prediction,
                    labels,
                    overlap=True,
                    frames_per_second=data.hparams_frames_per_second(hparams)))

          summary = sess.run(summary_op)
          summary_writer.add_summary(summary, sess.run(global_step))
          summary_writer.flush()

      if not write_summary_every_step:
        # Only write the summary variables for the final step.
        summary = sess.run(summary_op)
        summary_writer.add_summary(summary, sess.run(global_step))
        summary_writer.flush()


def run(config_map):
  """Run the infer script."""
  output_dir = os.path.expanduser(FLAGS.output_dir)

  config = config_map[FLAGS.config]
  hparams = config.hparams
  hparams.parse(FLAGS.hparams)

  # Batch size should always be 1 for inference.
  hparams.batch_size = 1

  if FLAGS.max_seconds_per_sequence:
    hparams.truncated_length_secs = FLAGS.max_seconds_per_sequence
  else:
    hparams.truncated_length_secs = 0

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

  if FLAGS.eval_loop:
    assert not FLAGS.checkpoint_path

    checkpoint_path = None
    while True:
      checkpoint_path = tf.contrib.training.wait_for_new_checkpoint(
          FLAGS.model_dir, last_checkpoint=checkpoint_path)
      model_inference(
          model_fn=config.model_fn,
          model_dir=FLAGS.model_dir,
          checkpoint_path=checkpoint_path,
          hparams=hparams,
          examples_path=FLAGS.examples_path,
          output_dir=output_dir,
          summary_writer=summary_writer,
          master=FLAGS.master,
          preprocess_examples=FLAGS.preprocess_examples,
          write_summary_every_step=False)
  else:
    model_inference(
        model_fn=config.model_fn,
        model_dir=FLAGS.model_dir,
        checkpoint_path=FLAGS.checkpoint_path,
        hparams=hparams,
        examples_path=FLAGS.examples_path,
        output_dir=output_dir,
        summary_writer=summary_writer,
        preprocess_examples=FLAGS.preprocess_examples,
        master=FLAGS.master)
