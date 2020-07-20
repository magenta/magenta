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

"""Trains a real-time arbitrary image stylization model for mobile device."""
import ast
import os

from magenta.models.arbitrary_image_stylization import arbitrary_image_stylization_build_mobilenet_model as build_mobilenet_model
from magenta.models.image_stylization import image_utils
from magenta.models.image_stylization import vgg
import tensorflow.compat.v1 as tf
import tf_slim as slim

DEFAULT_CONTENT_WEIGHTS = '{"vgg_16/conv3": 1}'
DEFAULT_STYLE_WEIGHTS = ('{"vgg_16/conv1": 0.5e-3, "vgg_16/conv2": 0.5e-3,'
                         ' "vgg_16/conv3": 0.5e-3, "vgg_16/conv4": 0.5e-3}')

flags = tf.app.flags
flags.DEFINE_float('clip_gradient_norm', 0, 'Clip gradients to this norm')
flags.DEFINE_float('learning_rate', 1e-5, 'Learning rate')
flags.DEFINE_float('total_variation_weight', 1e4, 'Total variation weight')
flags.DEFINE_string('content_weights', DEFAULT_CONTENT_WEIGHTS,
                    'Content weights')
flags.DEFINE_string('style_weights', DEFAULT_STYLE_WEIGHTS, 'Style weights')
flags.DEFINE_integer('batch_size', 8, 'Batch size.')
flags.DEFINE_integer(
    'image_size', 256, 'Image size in pixel. Style and content images are '
    'resized to image_size x image_size for training.')
flags.DEFINE_float('alpha', 0.25, 'Width Multiplier for the transform model')
flags.DEFINE_boolean('random_style_image_size', True,
                     'Whether to augment style images or not.')
flags.DEFINE_boolean(
    'augment_style_images', True,
    'Whether to resize the style images to a random size or not.')
flags.DEFINE_boolean('center_crop', False,
                     'Whether to center crop the style images.')
flags.DEFINE_integer(
    'ps_tasks', 0, 'Number of parameter servers. If 0, parameters '
    'are handled locally by the worker.')
flags.DEFINE_integer('save_summaries_secs', 15,
                     'Frequency at which summaries are saved, in seconds.')
flags.DEFINE_integer('save_interval_secs', 15,
                     'Frequency at which the model is saved, in seconds.')
flags.DEFINE_integer(
    'task', 0, 'Task ID. Used when training with multiple '
    'workers to identify each worker.')
flags.DEFINE_integer('train_steps', 8000000, 'Number of training steps.')
flags.DEFINE_string('master', '', 'BNS name of the TensorFlow master to use.')
flags.DEFINE_string('style_dataset_file', None, 'Style dataset file.')
flags.DEFINE_string('train_dir', None,
                    'Directory for checkpoints and summaries.')
flags.DEFINE_string('mobilenet_checkpoint', None,
                    'Path to the pre-trained mobilenet_v2 checkpoint.')

FLAGS = flags.FLAGS


def main(unused_argv=None):
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    # Forces all input processing onto CPU in order to reserve the GPU for the
    # forward inference and back-propagation.
    device = '/cpu:0' if not FLAGS.ps_tasks else '/job:worker/cpu:0'
    with tf.device(
        tf.train.replica_device_setter(FLAGS.ps_tasks, worker_device=device)):
      # Loads content images.
      content_inputs_, _ = image_utils.imagenet_inputs(FLAGS.batch_size,
                                                       FLAGS.image_size)

      # Loads style images.
      [style_inputs_, _, _] = image_utils.arbitrary_style_image_inputs(
          FLAGS.style_dataset_file,
          batch_size=FLAGS.batch_size,
          image_size=FLAGS.image_size,
          shuffle=True,
          center_crop=FLAGS.center_crop,
          augment_style_images=FLAGS.augment_style_images,
          random_style_image_size=FLAGS.random_style_image_size)

    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
      # Process style and content weight flags.
      content_weights = ast.literal_eval(FLAGS.content_weights)
      style_weights = ast.literal_eval(FLAGS.style_weights)

      # Define the model
      stylized_images, total_loss, loss_dict, \
            _ = build_mobilenet_model.build_mobilenet_model(
                content_inputs_,
                style_inputs_,
                mobilenet_trainable=False,
                style_params_trainable=True,
                transformer_trainable=True,
                mobilenet_end_point='layer_19',
                transformer_alpha=FLAGS.alpha,
                style_prediction_bottleneck=100,
                adds_losses=True,
                content_weights=content_weights,
                style_weights=style_weights,
                total_variation_weight=FLAGS.total_variation_weight,
            )

      # Adding scalar summaries to the tensorboard.
      for key in loss_dict:
        tf.summary.scalar(key, loss_dict[key])

      # Adding Image summaries to the tensorboard.
      tf.summary.image('image/0_content_inputs', content_inputs_, 3)
      tf.summary.image('image/1_style_inputs_aug', style_inputs_, 3)
      tf.summary.image('image/2_stylized_images', stylized_images, 3)

      # Set up training
      optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
      train_op = slim.learning.create_train_op(
          total_loss,
          optimizer,
          clip_gradient_norm=FLAGS.clip_gradient_norm,
          summarize_gradients=False)

      # Function to restore VGG16 parameters.
      init_fn_vgg = slim.assign_from_checkpoint_fn(vgg.checkpoint_file(),
                                                   slim.get_variables('vgg_16'))

      # Function to restore Mobilenet V2 parameters.
      mobilenet_variables_dict = {
          var.op.name: var for var in slim.get_model_variables('MobilenetV2')
      }
      init_fn_mobilenet = slim.assign_from_checkpoint_fn(
          FLAGS.mobilenet_checkpoint, mobilenet_variables_dict)

      # Function to restore VGG16 and Mobilenet V2 parameters.
      def init_sub_networks(session):
        init_fn_vgg(session)
        init_fn_mobilenet(session)

      # Run training
      slim.learning.train(
          train_op=train_op,
          logdir=os.path.expanduser(FLAGS.train_dir),
          master=FLAGS.master,
          is_chief=FLAGS.task == 0,
          number_of_steps=FLAGS.train_steps,
          init_fn=init_sub_networks,
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_interval_secs=FLAGS.save_interval_secs)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
