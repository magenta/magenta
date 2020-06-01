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
"""Evaluates the N-styles style transfer model."""

import ast
import os

from magenta.models.image_stylization import image_utils
from magenta.models.image_stylization import learning
from magenta.models.image_stylization import model
import tensorflow.compat.v1 as tf
import tf_slim as slim

DEFAULT_CONTENT_WEIGHTS = '{"vgg_16/conv3": 1.0}'
DEFAULT_STYLE_WEIGHTS = ('{"vgg_16/conv1": 1e-4, "vgg_16/conv2": 1e-4,'
                         ' "vgg_16/conv3": 1e-4, "vgg_16/conv4": 1e-4}')


flags = tf.app.flags
flags.DEFINE_boolean('style_grid', False,
                     'Whether to generate the style grid.')
flags.DEFINE_boolean('style_crossover', False,
                     'Whether to do a style crossover in the style grid.')
flags.DEFINE_boolean('learning_curves', True,
                     'Whether to evaluate learning curves for all styles.')
flags.DEFINE_integer('batch_size', 16, 'Batch size')
flags.DEFINE_integer('image_size', 256, 'Image size.')
flags.DEFINE_integer('eval_interval_secs', 60,
                     'Frequency, in seconds, at which evaluation is run.')
flags.DEFINE_integer('num_evals', 32, 'Number of evaluations of the losses.')
flags.DEFINE_integer('num_styles', None, 'Number of styles.')
flags.DEFINE_float('alpha', 1.0, 'Width multiplier')
flags.DEFINE_string('content_weights', DEFAULT_CONTENT_WEIGHTS,
                    'Content weights')
flags.DEFINE_string('eval_dir', None,
                    'Directory where the results are saved to.')
flags.DEFINE_string('train_dir', None,
                    'Directory for checkpoints and summaries')
flags.DEFINE_string('master', '',
                    'Name of the TensorFlow master to use.')
flags.DEFINE_string('style_coefficients', None,
                    'Scales the style weights conditioned on the style image.')
flags.DEFINE_string('style_dataset_file', None, 'Style dataset file.')
flags.DEFINE_string('style_weights', DEFAULT_STYLE_WEIGHTS,
                    'Style weights')
FLAGS = flags.FLAGS


def main(_):
  with tf.Graph().as_default():
    # Create inputs in [0, 1], as expected by vgg_16.
    inputs, _ = image_utils.imagenet_inputs(
        FLAGS.batch_size, FLAGS.image_size)
    evaluation_images = image_utils.load_evaluation_images(FLAGS.image_size)

    # Process style and weight flags
    if FLAGS.style_coefficients is None:
      style_coefficients = [1.0 for _ in range(FLAGS.num_styles)]
    else:
      style_coefficients = ast.literal_eval(FLAGS.style_coefficients)
    if len(style_coefficients) != FLAGS.num_styles:
      raise ValueError(
          'number of style coefficients differs from number of styles')
    content_weights = ast.literal_eval(FLAGS.content_weights)
    style_weights = ast.literal_eval(FLAGS.style_weights)

    # Load style images.
    style_images, labels, style_gram_matrices = image_utils.style_image_inputs(
        os.path.expanduser(FLAGS.style_dataset_file),
        batch_size=FLAGS.num_styles, image_size=FLAGS.image_size,
        square_crop=True, shuffle=False)
    labels = tf.unstack(labels)

    def _create_normalizer_params(style_label):
      """Creates normalizer parameters from a style label."""
      return {'labels': tf.expand_dims(style_label, 0),
              'num_categories': FLAGS.num_styles,
              'center': True,
              'scale': True}

    # Dummy call to simplify the reuse logic
    model.transform(
        inputs,
        alpha=FLAGS.alpha,
        reuse=False,
        normalizer_params=_create_normalizer_params(labels[0]))

    def _style_sweep(inputs):
      """Transfers all styles onto the input one at a time."""
      inputs = tf.expand_dims(inputs, 0)

      stylized_inputs = []
      for _, style_label in enumerate(labels):
        stylized_input = model.transform(
            inputs,
            alpha=FLAGS.alpha,
            reuse=True,
            normalizer_params=_create_normalizer_params(style_label))
        stylized_inputs.append(stylized_input)

      return tf.concat([inputs] + stylized_inputs, 0)

    if FLAGS.style_grid:
      style_row = tf.concat(
          [tf.ones([1, FLAGS.image_size, FLAGS.image_size, 3]), style_images],
          0)
      stylized_training_example = _style_sweep(inputs[0])
      stylized_evaluation_images = [
          _style_sweep(image) for image in tf.unstack(evaluation_images)]
      stylized_noise = _style_sweep(
          tf.random_uniform([FLAGS.image_size, FLAGS.image_size, 3]))
      stylized_style_images = [
          _style_sweep(image) for image in tf.unstack(style_images)]
      if FLAGS.style_crossover:
        grid = tf.concat(
            [style_row, stylized_training_example, stylized_noise] +
            stylized_evaluation_images + stylized_style_images,
            0)
      else:
        grid = tf.concat(
            [style_row, stylized_training_example, stylized_noise] +
            stylized_evaluation_images,
            0)
      if FLAGS.style_crossover:
        grid_shape = [
            3 + evaluation_images.get_shape().as_list()[0] + FLAGS.num_styles,
            1 + FLAGS.num_styles]
      else:
        grid_shape = [
            3 + evaluation_images.get_shape().as_list()[0],
            1 + FLAGS.num_styles]

      tf.summary.image(
          'Style Grid',
          tf.cast(
              image_utils.form_image_grid(
                  grid,
                  grid_shape,
                  [FLAGS.image_size, FLAGS.image_size],
                  3) * 255.0,
              tf.uint8))

    if FLAGS.learning_curves:
      metrics = {}
      for i, label in enumerate(labels):
        gram_matrices = dict(
            (key, value[i: i + 1])
            for key, value in style_gram_matrices.items())
        stylized_inputs = model.transform(
            inputs,
            alpha=FLAGS.alpha,
            reuse=True,
            normalizer_params=_create_normalizer_params(label))
        _, loss_dict = learning.total_loss(
            inputs, stylized_inputs, gram_matrices, content_weights,
            style_weights, reuse=i > 0)
        for key, value in loss_dict.items():
          metrics['{}_style_{}'.format(key, i)] = slim.metrics.streaming_mean(
              value)

      names_values, names_updates = slim.metrics.aggregate_metric_map(metrics)
      for name, value in names_values.items():
        summary_op = tf.summary.scalar(name, value, [])
        print_op = tf.Print(summary_op, [value], name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, print_op)
      eval_op = list(names_updates.values())
      num_evals = FLAGS.num_evals
    else:
      eval_op = None
      num_evals = 1

    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=os.path.expanduser(FLAGS.train_dir),
        logdir=os.path.expanduser(FLAGS.eval_dir),
        eval_op=eval_op,
        num_evals=num_evals,
        eval_interval_secs=FLAGS.eval_interval_secs)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
