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

"""Genarates a TF-Hub module for arbitrary image stylization.

The module is compatible with TF-2 and is intended as a demonstration of TF-Hub
module creation based on TF-1 models.

The created hub module can be used for image stylization with:
  m = hub.load('hub_handle')
  stylized_image = m(content_image, style_image)

where content_image, style_image and the generated stylized_image are 4d arrays,
with the first being the batch dimension, that can be 1 for individual images.
The input and output values of the images should be in the range [0, 1].
The shapes of content and style image don't have to match. Output image shape
is the same as the content image shape.

A pre-trained checkpoint for the given model is available at
https://storage.googleapis.com/download.magenta.tensorflow.org/models/arbitrary_style_transfer.tar.gz
One can download and extract this tar file and to provide the path to the
checkpoint in it to the checkpoint flag.
"""
from absl import flags

from magenta.models.arbitrary_image_stylization import arbitrary_image_stylization_build_model
import tensorflow.compat.v1 as tf

flags.DEFINE_string('checkpoint', None, 'Path to the model checkpoint.')
flags.DEFINE_string('export_path', None, 'Path where to save the hub module.')
FLAGS = flags.FLAGS


def build_network(content_img, style_img):
  """Builds the neural network for image stylization."""
  stylize_op, _, _, _ = arbitrary_image_stylization_build_model.build_model(
      content_img,
      style_img,
      trainable=False,
      is_training=False,
      adds_losses=False)
  return stylize_op


def get_stylize_fn():
  """Creates a tf.function for stylization."""
  input_spec = [
      tf.TensorSpec((None, None, None, 3), tf.float32),
      tf.TensorSpec((None, None, None, 3), tf.float32)
  ]
  predict_feeds = []
  predict_fetches = []

  def umbrella_function(content_img, style_img):
    predict_feeds.extend([content_img, style_img])
    predict_result = build_network(content_img, style_img)
    predict_fetches.extend([
        predict_result,
    ])
    return predict_result

  umbrella_wrapped = tf.compat.v1.wrap_function(umbrella_function, input_spec)
  fn = umbrella_wrapped.prune(predict_feeds, predict_fetches)
  return fn


def create_hub_module_object():
  """Creates an exportable saved model object."""
  obj = tf.train.Checkpoint()
  obj.__call__ = get_stylize_fn()
  obj.variables = list(obj.__call__.graph.variables)
  # To avoid error related to reading expected variable save_counter.
  obj.save_counter  # pylint: disable=pointless-statement
  return obj


def main(unused_argv=None):
  obj = create_hub_module_object()
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    tf.train.Saver(obj.variables).restore(sess, FLAGS.checkpoint)
    tf.saved_model.save(obj, FLAGS.export_path, signatures=obj.__call__)
  tf.logging.info('Saved hub module in: %s', FLAGS.export_path)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.compat.v1.app.run(main)


if __name__ == '__main__':
  console_entry_point()
