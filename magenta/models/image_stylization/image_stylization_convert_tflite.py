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

"""Convert trained mobile style transfer model to TF Lite."""
import os
import tempfile

from magenta.models.image_stylization import model
from magenta.models.image_stylization import ops
import tensorflow.compat.v1 as tf

flags = tf.flags
flags.DEFINE_integer('num_styles', 1,
                     'Number of styles the model was trained on.')
flags.DEFINE_float('alpha', 1.0,
                   'Width multiplier the model was trained on.')
flags.DEFINE_string('checkpoint', None, 'Checkpoint to load the model from')
flags.DEFINE_string('output_model', None, 'Output TensorFlow Lite model.')
flags.DEFINE_bool('quantize', False,
                  'Whether to quantize the TensorFlow Lite model.')
flags.DEFINE_integer(
    'image_size', 384,
    'Default input image size in pixel for the generated TensorFlow Lite model.'
    'model. The model will take image_size x image_size RGB image as input.')
FLAGS = flags.FLAGS


def _load_checkpoint(sess, checkpoint):
  """Loads a checkpoint file into the session."""
  model_saver = tf.train.Saver(tf.global_variables())
  checkpoint = os.path.expanduser(checkpoint)
  if tf.gfile.IsDirectory(checkpoint):
    checkpoint = tf.train.latest_checkpoint(checkpoint)
    tf.logging.info('loading latest checkpoint file: {}'.format(checkpoint))
  model_saver.restore(sess, checkpoint)


def _export_to_saved_model(checkpoint, alpha, num_styles):
  """Export a image stylization checkpoint to SavedModel format."""
  saved_model_dir = tempfile.mkdtemp()

  with tf.Graph().as_default(), tf.Session() as sess:
    # Define input tensor as placeholder to allow export
    input_image_tensor = tf.placeholder(
        tf.float32, shape=(1, None, None, 3), name='input_image')
    weights_tensor = tf.placeholder(
        tf.float32, shape=num_styles, name='style_weights')

    # Load the graph definition from Magenta
    stylized_image_tensor = model.transform(
        input_image_tensor,
        alpha=alpha,
        normalizer_fn=ops.weighted_instance_norm,
        normalizer_params={
            'weights': weights_tensor,
            'num_categories': FLAGS.num_styles,
            'center': True,
            'scale': True
        })

    # Load model weights from downloaded checkpoint file
    _load_checkpoint(sess, checkpoint)

    # Write SavedModel for serving or conversion to TF Lite
    tf.saved_model.simple_save(
        sess,
        saved_model_dir,
        inputs={
            input_image_tensor.name: input_image_tensor,
            weights_tensor.name: weights_tensor
        },
        outputs={'stylized_image': stylized_image_tensor})

  return saved_model_dir


def _convert_to_tflite(saved_model_dir, num_styles, image_size, quantize,
                       output_model):
  """Convert a image stylization saved model to TensorFlow Lite format."""
  # Append filename if output_model is a directory name
  if tf.io.gfile.isdir(output_model):
    if quantize:
      filename = 'stylize_quantized.tflite'
    else:
      filename = 'stylize.tflite'
    output_model = os.path.join(output_model, filename)

  # Initialize TF Lite Converter
  converter = tf.lite.TFLiteConverter.from_saved_model(
      saved_model_dir=saved_model_dir,
      input_shapes={
          'input_image': [None, image_size, image_size, 3],
          'style_weights': num_styles
      })

  # Specify quantization option
  if quantize:
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

  # Convert and save the TF Lite model
  tflite_model = converter.convert()
  with tf.io.gfile.GFile(output_model, 'wb') as f:
    f.write(tflite_model)
  tf.logging.info('Converted to TF Lite model: %s; Size: %d KB.' %
                  (output_model, len(tflite_model) / 1024))


def main(unused_argv=None):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Load model weights from trained checkpoint and export to SavedModel format
  saved_model_dir = _export_to_saved_model(
      os.path.expanduser(FLAGS.checkpoint), FLAGS.alpha, FLAGS.num_styles)

  # Convert from SavedModel to TensorFlow Lite format
  _convert_to_tflite(saved_model_dir, FLAGS.num_styles, FLAGS.image_size,
                     FLAGS.quantize, os.path.expanduser(FLAGS.output_model))


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
