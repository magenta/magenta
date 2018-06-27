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

"""Utilities for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports

from . import data
from . import model
from .infer_util import pianoroll_to_note_sequence
from .infer_util import sequence_to_valued_intervals

from mir_eval.transcription import precision_recall_f1_overlap

import pretty_midi
import tensorflow as tf
import tensorflow.contrib.slim as slim


def _get_data(examples_path, hparams, is_training):
  hparams_dict = hparams.values()
  batch, _ = data.provide_batch(
      hparams.batch_size,
      examples=examples_path,
      hparams=hparams,
      truncated_length=hparams_dict.get('truncated_length', None),
      is_training=is_training)
  return batch


# Should not be called from within the graph to avoid redundant summaries.
def _trial_summary(hparams, examples_path, output_dir):
  """Writes a tensorboard text summary of the trial."""

  examples_path_summary = tf.summary.text(
      'examples_path', tf.constant(examples_path, name='examples_path'),
      collections=[])

  tf.logging.info('Writing hparams summary: %s', hparams)

  hparams_dict = hparams.values()

  # Create a markdown table from hparams.
  header = '| Key | Value |\n| :--- | :--- |\n'
  keys = sorted(hparams_dict.keys())
  lines = ['| %s | %s |' % (key, str(hparams_dict[key])) for key in keys]
  hparams_table = header + '\n'.join(lines) + '\n'

  hparam_summary = tf.summary.text(
      'hparams', tf.constant(hparams_table, name='hparams'), collections=[])

  with tf.Session() as sess:
    writer = tf.summary.FileWriter(output_dir, graph=sess.graph)
    writer.add_summary(examples_path_summary.eval())
    writer.add_summary(hparam_summary.eval())
    writer.close()


def train(train_dir,
          examples_path,
          hparams,
          checkpoints_to_keep=5,
          keep_checkpoint_every_n_hours=1,
          num_steps=None):
  """Train loop."""
  tf.gfile.MakeDirs(train_dir)

  _trial_summary(hparams, examples_path, train_dir)
  with tf.Graph().as_default():
    transcription_data = _get_data(examples_path, hparams, is_training=True)

    loss, losses, unused_labels, unused_predictions, images = model.get_model(
        transcription_data, hparams, is_training=True)

    tf.summary.scalar('loss', loss)
    for label, loss_collection in losses.iteritems():
      loss_label = 'losses/' + label
      tf.summary.scalar(loss_label, tf.reduce_mean(loss_collection))
    for name, image in images.iteritems():
      tf.summary.image(name, image)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
        hparams.learning_rate,
        global_step,
        hparams.decay_steps,
        hparams.decay_rate,
        staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_op = slim.learning.create_train_op(
        loss,
        optimizer,
        clip_gradient_norm=hparams.clip_norm,
        summarize_gradients=True)

    logging_dict = {'global_step': tf.train.get_global_step(), 'loss': loss}

    hooks = [tf.train.LoggingTensorHook(logging_dict, every_n_iter=100)]
    if num_steps:
      hooks.append(tf.train.StopAtStepHook(num_steps))

    scaffold = tf.train.Scaffold(
        saver=tf.train.Saver(
            max_to_keep=checkpoints_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours))

    tf.contrib.training.train(
        train_op=train_op,
        logdir=train_dir,
        scaffold=scaffold,
        hooks=hooks,
        save_checkpoint_secs=300)


def evaluate(train_dir,
             eval_dir,
             examples_path,
             hparams,
             num_batches=None):
  """Evaluate the model repeatedly."""
  tf.gfile.MakeDirs(eval_dir)

  _trial_summary(hparams, examples_path, eval_dir)
  with tf.Graph().as_default():
    transcription_data = _get_data(examples_path, hparams, is_training=False)
    unused_loss, losses, labels, predictions, images = model.get_model(
        transcription_data, hparams, is_training=False)

    _, metrics_to_updates = _get_eval_metrics(
        losses, labels, predictions, images, hparams)

    hooks = [
        tf.contrib.training.StopAfterNEvalsHook(
            num_batches or transcription_data.num_batches),
        tf.contrib.training.SummaryAtEndHook(eval_dir)]
    tf.contrib.training.evaluate_repeatedly(
        train_dir,
        eval_ops=metrics_to_updates.values(),
        hooks=hooks,
        eval_interval_secs=60,
        timeout=None)


def test(checkpoint_path, test_dir, examples_path, hparams,
         num_batches=None):
  """Evaluate the model at a single checkpoint."""
  tf.gfile.MakeDirs(test_dir)

  _trial_summary(hparams, examples_path, test_dir)
  with tf.Graph().as_default():
    transcription_data = _get_data(
        examples_path, hparams, is_training=False)
    unused_loss, losses, labels, predictions, images = model.get_model(
        transcription_data, hparams, is_training=False)

    metrics_to_values, metrics_to_updates = _get_eval_metrics(
        losses, labels, predictions, images, hparams)

    metric_values = slim.evaluation.evaluate_once(
        checkpoint_path=checkpoint_path,
        logdir=test_dir,
        num_evals=num_batches or transcription_data.num_batches,
        eval_op=metrics_to_updates.values(),
        final_op=metrics_to_values.values())

    metrics_to_values = dict(zip(metrics_to_values.keys(), metric_values))
    for metric in metrics_to_values:
      print('%s: %f' % (metric, metrics_to_values[metric]))


def _note_metrics_op(labels, predictions, hparams, offset_ratio=None):
  """An op that provides access to mir_eval note scores through a py_func."""

  def _note_metrics(labels, predictions):
    """A pyfunc that wraps a call to precision_recall_f1_overlap."""
    est_sequence = pianoroll_to_note_sequence(
        predictions,
        frames_per_second=data.hparams_frames_per_second(hparams),
        min_duration_ms=hparams.min_duration_ms)

    ref_sequence = pianoroll_to_note_sequence(
        labels,
        frames_per_second=data.hparams_frames_per_second(hparams),
        min_duration_ms=hparams.min_duration_ms)

    est_intervals, est_pitches = sequence_to_valued_intervals(
        est_sequence, hparams.min_duration_ms)
    ref_intervals, ref_pitches = sequence_to_valued_intervals(
        ref_sequence, hparams.min_duration_ms)

    if est_intervals.size == 0 or ref_intervals.size == 0:
      return 0., 0., 0.
    note_precision, note_recall, note_f1, _ = precision_recall_f1_overlap(
        ref_intervals,
        pretty_midi.note_number_to_hz(ref_pitches),
        est_intervals,
        pretty_midi.note_number_to_hz(est_pitches),
        offset_ratio=offset_ratio)

    return note_precision, note_recall, note_f1

  note_precision, note_recall, note_f1 = tf.py_func(
      _note_metrics, [labels, predictions],
      [tf.float64, tf.float64, tf.float64],
      name='note_scores')

  return note_precision, note_recall, note_f1


def _get_eval_metrics(losses, labels, predictions, images, hparams):
  """Returns evaluation metrics.

  Args:
    losses: a dict containing losses with a training job.
    labels: a numpy array or a dict. If a dict, it contains
      multiple labels for different tasks.
    predictions: a numpy array or a dict. The type of predictions
      must match that of labels. If both are dicts, they must have
      the same keys.
    images: a dict of images.
    hparams: a set of hyperparameters.

  Returns: metrics to evaluate and update.
  """
  image_prefix = 'images/'
  if not isinstance(labels, dict):
    labels = {'default': labels}
    predictions = {'default': predictions}

  metric_map = {}

  def expand_key(key, metric_name, size):
    """Return expanded metric name based on size."""
    if size > 1:
      return 'metrics/%s/%s' % (key, metric_name)
    else:
      return 'metrics/%s' % (metric_name)

  size = len(labels)
  for key in labels.keys():
    metric_map[expand_key(key, 'accuracy', size)] = tf.metrics.accuracy(
        labels[key], predictions[key])
    metric_map[expand_key(key, 'precision', size)] = tf.metrics.precision(
        labels[key], predictions[key])
    metric_map[expand_key(key, 'recall', size)] = tf.metrics.recall(
        labels[key], predictions[key])
    metric_map[expand_key(key, 'true_positives',
                          size)] = tf.metrics.true_positives(
                              labels[key], predictions[key])
    metric_map[expand_key(key, 'false_positives',
                          size)] = tf.metrics.false_positives(
                              labels[key], predictions[key])
    metric_map[expand_key(key, 'false_negatives',
                          size)] = tf.metrics.false_negatives(
                              labels[key], predictions[key])
    metric_map[expand_key(key, 'roc', size)] = tf.metrics.auc(
        labels[key], predictions[key])

    # these metrics might be meaningless in the windowed case
    note_precision, note_recall, note_f1 = _note_metrics_op(
        labels[key], predictions[key], hparams)
    metric_map[expand_key(key, 'note_precision',
                          size)] = tf.metrics.mean(note_precision)
    metric_map[expand_key(key, 'note_recall',
                          size)] = tf.metrics.mean(note_recall)
    metric_map[expand_key(key, 'note_f1', size)] = tf.metrics.mean(note_f1)

    note_tuple = _note_metrics_op(labels[key], predictions[key], hparams, .2)
    note_precision_with_offsets = note_tuple[0]
    note_recall_with_offsets = note_tuple[1]
    note_f1_with_offsets = note_tuple[2]
    metric_map[expand_key(key, 'note_precision_with_offsets',
                          size)] = tf.metrics.mean(note_precision_with_offsets)
    metric_map[expand_key(key, 'note_recall_with_offsets',
                          size)] = tf.metrics.mean(note_recall_with_offsets)
    metric_map[expand_key(key, 'note_f1_with_offsets',
                          size)] = tf.metrics.mean(note_f1_with_offsets)

    try:
      onset_labels = tf.get_default_graph().get_tensor_by_name(
          'onsets/onset_labels_flat:0')
      onset_predictions = tf.get_default_graph().get_tensor_by_name(
          'onsets/onset_predictions_flat:0')
      onset_note_precision, onset_note_recall, onset_note_f1 = _note_metrics_op(
          onset_labels, onset_predictions, hparams)
      metric_map[expand_key(key, 'onset_note_precision',
                            size)] = tf.metrics.mean(onset_note_precision)
      metric_map[expand_key(key, 'onset_note_recall',
                            size)] = tf.metrics.mean(onset_note_recall)
      metric_map[expand_key(key, 'onset_note_f1',
                            size)] = tf.metrics.mean(onset_note_f1)
    except KeyError:
      # no big deal if we can't find the tensors
      pass

  # Create a local variable to store the last batch of images.
  for image_name, image in images.iteritems():
    var_name = image_prefix + image_name
    with tf.variable_scope(image_name, values=[image]):
      local_image = tf.Variable(
          initial_value=tf.zeros(
              [1 if d is None else d for d in image.shape.as_list()],
              image.dtype),
          name=var_name,
          trainable=False,
          collections=[tf.GraphKeys.LOCAL_VARIABLES],
          validate_shape=False)
    metric_map[var_name] = (
        local_image, tf.assign(local_image, image, validate_shape=False))

  # Calculate streaming means for each of the losses.
  loss_labels = []
  for label, loss_collection in losses.iteritems():
    loss_label = 'losses/' + label
    loss_labels.append(loss_label)
    metric_map[loss_label] = tf.metrics.mean(loss_collection)

  metrics_to_values, metrics_to_updates = (
      tf.contrib.metrics.aggregate_metric_map(metric_map))

  for metric_name, metric_value in metrics_to_values.iteritems():
    if metric_name.startswith(image_prefix):
      tf.summary.image(metric_name[len(image_prefix):], metric_value)
    else:
      tf.summary.scalar(metric_name, metric_value)

  # Calculate total loss metric by adding up all the individual loss means.
  total_loss = tf.add_n([metrics_to_values[l] for l in loss_labels])
  metrics_to_values['loss'] = total_loss
  tf.summary.scalar('loss', total_loss)

  for key in labels.keys():
    # Calculate F1 Score based on precision and recall.
    precision = metrics_to_values[expand_key(key, 'precision', size)]
    recall = metrics_to_values[expand_key(key, 'recall', size)]

    f1_score = tf.where(
        tf.greater(precision + recall, 0),
        2 * ((precision * recall) / (precision + recall)), 0)
    metrics_to_values[expand_key(key, 'f1_score', size)] = f1_score
    tf.summary.scalar(expand_key(key, 'f1_score', size), f1_score)

    # Calculate accuracy without true negatives.
    true_positives = metrics_to_values[expand_key(key, 'true_positives', size)]
    false_positives = metrics_to_values[expand_key(key, 'false_positives',
                                                   size)]
    false_negatives = metrics_to_values[expand_key(key, 'false_negatives',
                                                   size)]
    accuracy_without_true_negatives = tf.where(
        tf.greater(true_positives + false_positives + false_negatives,
                   0), true_positives /
        (true_positives + false_positives + false_negatives), 0)
    metrics_to_values[expand_key(key, 'accuracy_without_true_negatives',
                                 size)] = (accuracy_without_true_negatives)
    tf.summary.scalar(
        expand_key(key, 'accuracy_without_true_negatives', size),
        accuracy_without_true_negatives)

  return metrics_to_values, metrics_to_updates
