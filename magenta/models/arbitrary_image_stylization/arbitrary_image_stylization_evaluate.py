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
"""Evaluates a real-time arbitrary image stylization model.

For example of usage see README.md.
"""
import ast

from magenta.models.arbitrary_image_stylization import arbitrary_image_stylization_build_model as build_model
from magenta.models.image_stylization import image_utils
import tensorflow.compat.v1 as tf
import tf_slim as slim

DEFAULT_CONTENT_WEIGHTS = '{"vgg_16/conv3": 1.0}'
DEFAULT_STYLE_WEIGHTS = ('{"vgg_16/conv1": 1e-3, "vgg_16/conv2": 1e-3,'
                         ' "vgg_16/conv3": 1e-3, "vgg_16/conv4": 1e-3}')

flags = tf.app.flags
flags.DEFINE_float('total_variation_weight', 1e4, 'Total variation weight')
flags.DEFINE_string('content_weights', DEFAULT_CONTENT_WEIGHTS,
                    'Content weights')
flags.DEFINE_string('style_weights', DEFAULT_STYLE_WEIGHTS, 'Style weights')
flags.DEFINE_integer('batch_size', 16, 'Batch size')
flags.DEFINE_integer('image_size', 256, 'Image size.')
flags.DEFINE_integer('eval_interval_secs', 60,
                     'Frequency, in seconds, at which evaluation is run.')
flags.DEFINE_integer('num_evaluation_styles', 1024,
                     'Total number of evaluation styles.')
flags.DEFINE_string('eval_dir', None,
                    'Directory where the results are saved to.')
flags.DEFINE_string('checkpoint_dir', None,
                    'Directory for checkpoints and summaries')
flags.DEFINE_string('master', '', 'BNS name of the TensorFlow master to use.')
flags.DEFINE_string('eval_name', 'eval', 'Name of evaluation.')
flags.DEFINE_string('eval_style_dataset_file', None, 'path to the evaluation'
                    'style dataset file.')
FLAGS = flags.FLAGS


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    # Loads content images.
    eval_content_inputs_, _ = image_utils.imagenet_inputs(
        FLAGS.batch_size, FLAGS.image_size)

    # Process style and content weight flags.
    content_weights = ast.literal_eval(FLAGS.content_weights)
    style_weights = ast.literal_eval(FLAGS.style_weights)

    # Loads evaluation style images.
    eval_style_inputs_, _, _ = image_utils.arbitrary_style_image_inputs(
        FLAGS.eval_style_dataset_file,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.image_size,
        center_crop=True,
        shuffle=True,
        augment_style_images=False,
        random_style_image_size=False)

    # Computes stylized noise.
    stylized_noise, _, _, _ = build_model.build_model(
        tf.random_uniform(
            [min(4, FLAGS.batch_size), FLAGS.image_size, FLAGS.image_size, 3]),
        tf.slice(eval_style_inputs_, [0, 0, 0, 0],
                 [min(4, FLAGS.batch_size), -1, -1, -1]),
        trainable=False,
        is_training=False,
        reuse=None,
        inception_end_point='Mixed_6e',
        style_prediction_bottleneck=100,
        adds_losses=False)

    # Computes stylized images.
    stylized_images, _, loss_dict, _ = build_model.build_model(
        eval_content_inputs_,
        eval_style_inputs_,
        trainable=False,
        is_training=False,
        reuse=True,
        inception_end_point='Mixed_6e',
        style_prediction_bottleneck=100,
        adds_losses=True,
        content_weights=content_weights,
        style_weights=style_weights,
        total_variation_weight=FLAGS.total_variation_weight)

    # Adds Image summaries to the tensorboard.
    tf.summary.image('image/{}/0_eval_content_inputs'.format(FLAGS.eval_name),
                     eval_content_inputs_, 3)
    tf.summary.image('image/{}/1_eval_style_inputs'.format(FLAGS.eval_name),
                     eval_style_inputs_, 3)
    tf.summary.image('image/{}/2_eval_stylized_images'.format(FLAGS.eval_name),
                     stylized_images, 3)
    tf.summary.image('image/{}/3_stylized_noise'.format(FLAGS.eval_name),
                     stylized_noise, 3)

    metrics = {}
    for key, value in loss_dict.items():
      metrics[key] = tf.metrics.mean(value)

    names_values, names_updates = slim.metrics.aggregate_metric_map(metrics)
    for name, value in names_values.items():
      slim.summaries.add_scalar_summary(value, name, print_summary=True)
    eval_op = list(names_updates.values())
    num_evals = FLAGS.num_evaluation_styles / FLAGS.batch_size

    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_dir,
        eval_op=eval_op,
        num_evals=num_evals,
        eval_interval_secs=FLAGS.eval_interval_secs)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)

if __name__ == '__main__':
  console_entry_point()
