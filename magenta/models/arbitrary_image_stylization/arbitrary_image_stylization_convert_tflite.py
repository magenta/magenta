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

"""Convert trained mobile arbitrary style transfer model to TF Lite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from magenta.models.arbitrary_image_stylization \
  import arbitrary_image_stylization_build_mobilenet_model as build_mobilenet_model
import numpy as np
import tensorflow.compat.v1 as tf

flags = tf.flags
flags.DEFINE_float('alpha', 0.25, 'Width multiplier of the transform model.')
flags.DEFINE_string('checkpoint', None, 'Checkpoint to load the model from')
flags.mark_flag_as_required('checkpoint')
flags.DEFINE_string('output_dir', None,
                    'Directory to export TensorFlow Lite models.')
flags.mark_flag_as_required('output_dir')
flags.DEFINE_string('style_dataset_file', None,
                    'Style dataset file used for quantization.')
flags.DEFINE_string('imagenet_data_dir', None,
                    'Path to the ImageNet data used for quantization.')
flags.DEFINE_integer(
    'image_size', 384,
    'Default input image size for the generated TensorFlow Lite models.')
FLAGS = flags.FLAGS


def load_checkpoint(sess, checkpoint):
  """Loads a checkpoint file into the session.

  Args:
    sess: tf.Session, the TF session to load variables from the checkpoint to.
    checkpoint: str, path to the checkpoint file.
  """
  model_saver = tf.train.Saver(tf.global_variables())
  checkpoint = os.path.expanduser(checkpoint)
  if tf.gfile.IsDirectory(checkpoint):
    checkpoint = tf.train.latest_checkpoint(checkpoint)
    tf.logging.info('loading latest checkpoint file: {}'.format(checkpoint))
  model_saver.restore(sess, checkpoint)


def export_to_saved_model(checkpoint, alpha):
  """Export arbitrary style transfer trained checkpoints to SavedModel format.

  Args:
    checkpoint: str, path to the checkpoint file.
    alpha: Width Multiplier of the transform model.

  Returns:
    (str, str) Path to the exported style predict and style transform
    SavedModel.
  """
  saved_model_dir = tempfile.mkdtemp()
  predict_saved_model_folder = os.path.join(saved_model_dir, 'predict')
  transform_saved_model_folder = os.path.join(saved_model_dir, 'transform')

  with tf.Graph().as_default(), tf.Session() as sess:
    # Defines place holder for the style image.
    style_image_tensor = tf.placeholder(
        tf.float32, shape=[None, None, None, 3], name='style_image')

    # Defines place holder for the content image.
    content_image_tensor = tf.placeholder(
        tf.float32, shape=[None, None, None, 3], name='content_image')

    # Defines the model.
    stylized_images, _, _, bottleneck_feat = \
      build_mobilenet_model.build_mobilenet_model(
          content_image_tensor,
          style_image_tensor,
          mobilenet_trainable=False,
          style_params_trainable=False,
          transformer_trainable=False,
          style_prediction_bottleneck=100,
          transformer_alpha=alpha,
          mobilenet_end_point='layer_19',
          adds_losses=False)

    # Load model weights from  checkpoint file
    load_checkpoint(sess, checkpoint)

    # Write SavedModel for serving or conversion to TF Lite
    tf.saved_model.simple_save(
        sess,
        predict_saved_model_folder,
        inputs={style_image_tensor.name: style_image_tensor},
        outputs={'style_bottleneck': bottleneck_feat})
    tf.logging.debug('Export predict SavedModel to', predict_saved_model_folder)

    tf.saved_model.simple_save(
        sess,
        transform_saved_model_folder,
        inputs={
            content_image_tensor.name: content_image_tensor,
            'style_bottleneck': bottleneck_feat
        },
        outputs={'stylized_image': stylized_images})
    tf.logging.debug('Export transform SavedModel to',
                     transform_saved_model_folder)

  return predict_saved_model_folder, transform_saved_model_folder


def convert_saved_model_to_tflite(saved_model_dir, input_shapes,
                                  float_tflite_file, quantized_tflite_file):
  """Convert SavedModel to TF Lite format.

  Also apply weight quantization to generate quantized model

  Args:
    saved_model_dir: str, path to the SavedModel directory.
    input_shapes: dict, input shapes of the SavedModel.
    float_tflite_file: str, path to export the float TF Lite model.
    quantized_tflite_file: str, path to export the weight quantized TF Lite
      model.
  Returns: (str, str) Path to the exported style predict and style transform
    SavedModel.
  """

  converter = tf.lite.TFLiteConverter.from_saved_model(
      saved_model_dir=saved_model_dir, input_shapes=input_shapes)

  tflite_float_model = converter.convert()
  with tf.gfile.GFile(float_tflite_file, 'wb') as f:
    f.write(tflite_float_model)

  tf.logging.info('Converted to TF Lite float model: %s; Size: %d KB.' %
                  (float_tflite_file, len(tflite_float_model) / 1024))

  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_quantize_model = converter.convert()
  with tf.gfile.GFile(quantized_tflite_file, 'wb') as f:
    f.write(tflite_quantize_model)

  tf.logging.info(
      'Converted to TF Lite weight quantized model: %s; Size: %d KB.' %
      (quantized_tflite_file, len(tflite_quantize_model) / 1024))


def convert_saved_model_to_int8_tflite(saved_model_dir, input_shapes,
                                       representative_dataset,
                                       calibrated_tflite_file):
  """Convert SavedModel to int8 quantized TF Lite format.

  Args:
    saved_model_dir: str, path to the SavedModel directory.
    input_shapes: dict, input shapes of the SavedModel.
    representative_dataset: function, generator function to feed to
      TFLiteConverter
    calibrated_tflite_file: str, path to export the full int8 quantized TF Lite
      model.
  Returns: (str, str) Path to the exported style predict and style transform
    SavedModel.
  """

  converter = tf.lite.TFLiteConverter.from_saved_model(
      saved_model_dir=saved_model_dir, input_shapes=input_shapes)

  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_dataset

  tflite_full_int8_quantize_model = converter.convert()
  with tf.gfile.GFile(calibrated_tflite_file, 'wb') as f:
    f.write(tflite_full_int8_quantize_model)

  tf.logging.info(
      'Converted to TF Lite full integer quantized model: %s; Size: %d KB.' %
      (calibrated_tflite_file, len(tflite_full_int8_quantize_model) / 1024))


def parse_function(image_size, raw_image_key_name):
  """Generate parse function for parsing the TFRecord training dataset.

  Read the image example and resize it to desired size.

  Args:
    image_size: int, target size to resize the image to
    raw_image_key_name: str, name of the JPEG image in each TFRecord entry

  Returns:
    A map function to use with tf.data.Dataset.map() .
  """

  def func(example_proto):
    """A generator to be used as representative_dataset for TFLiteConverter."""
    image_raw = tf.io.parse_single_example(
        example_proto,
        features={raw_image_key_name: tf.FixedLenFeature([], tf.string)},
    )
    image = tf.image.decode_jpeg(image_raw[raw_image_key_name])
    image = tf.expand_dims(image, axis=0)
    image = tf.image.resize_bilinear(image, (image_size, image_size))
    image = tf.squeeze(image, axis=0)
    image = image / 255.0
    return image

  return func


def rgb_filter_function(image):
  """Filter function that only lets RGB images go through.

  Args:
    image: Tensor, the image tensor to be checked.
  Returns: bool, whether to keep the image or not.
  """
  shape = tf.shape(image)[2]
  return tf.math.equal(shape, 3)


def get_calibration_dataset(image_size, style_dataset_file, imagenet_data_dir):
  """Generate calibration dataset from the training dataset.

  Args:
    image_size: int, image size to resize the training images to.
    style_dataset_file: str, path to the style images TFRecord file.
    imagenet_data_dir: str, path to the ImageNet data.

  Returns:
    (str, str) Path to the style dataset and content dataset for calibration.
  """

  # Load style dataset
  style_dataset = tf.data.TFRecordDataset(style_dataset_file)
  style_dataset = style_dataset.map(parse_function(image_size, 'image_raw'),
                                    num_parallel_calls= \
                                    tf.data.experimental.AUTOTUNE)
  style_dataset = style_dataset.filter(rgb_filter_function)

  # Load content dataset
  tf_record_pattern = os.path.join(imagenet_data_dir, '%s-*' % 'train')
  data_files = tf.gfile.Glob(tf_record_pattern)
  content_dataset = tf.data.TFRecordDataset(data_files)
  content_dataset = content_dataset.map(
      parse_function(image_size, 'image/encoded'))
  content_dataset = content_dataset.filter(rgb_filter_function)

  return style_dataset, content_dataset


def predict_model_gen(session, style_dataset, sample_count):
  """Create a generator function that emits style images.

  Args:
    session: tf.Session, the session that contains subgraph to load the traning
      dataset
    style_dataset: tf.data.Dataset that contains training style images.
    sample_count: int, number of sample to create.

  Returns:
    (str, str) A generator function to use as representative dataset for
    TFLiteConverter.
  """

  def generator():
    dataset = style_dataset.batch(1)
    iterator = tf.data.make_initializable_iterator(dataset)
    session.run(iterator.initializer)
    next_element = iterator.get_next()
    for _ in range(sample_count):
      input_value = session.run(next_element)
      yield [input_value]

  return generator


def calculate_style_bottleneck(session,
                               predict_saved_model,
                               style_dataset,
                               batch_size=10,
                               min_sample_count=30):
  """Calculate style bottleneck using style predict SavedModel.

  Args:
    session: tf.Session, the session that contains subgraph to load the traning
      dataset.
    predict_saved_model: str, path to the style predict SavedModel.
    style_dataset: tf.data.Dataset that contains training style images.
    batch_size: int, Batch size used when calculating style bottlenecks.
    min_sample_count: int, the minimum number of sample needed.

  Returns:
    [[[int]]] Value of style bottleneck for the style images being processed.
  """

  styles_bottleneck = None

  tf.logging.info('Started calculating style bottlenecks.')

  # Load and run the style predict SavedModel
  loaded = tf.saved_model.loader.load(session, ['serve'], predict_saved_model)
  bottleneck_tensor_name = loaded.signature_def['serving_default'].outputs[
      'style_bottleneck'].name

  # Load the style dataset
  dataset = style_dataset.batch(batch_size)
  iterator = tf.data.make_initializable_iterator(dataset)
  session.run(iterator.initializer)
  next_element = iterator.get_next()

  for _ in range(np.math.ceil(min_sample_count / batch_size)):
    style_image = session.run(next_element)

    style_bottleneck_batch = session.run(
        bottleneck_tensor_name, feed_dict={'style_image:0': style_image})
    if styles_bottleneck is None:
      styles_bottleneck = style_bottleneck_batch
    else:
      styles_bottleneck = np.append(
          styles_bottleneck, style_bottleneck_batch, axis=0)

  tf.logging.info('Finished calculating style bottlenecks.')

  return styles_bottleneck


def transform_model_gen(session, predict_saved_model, style_dataset,
                        content_dataset, sample_count):
  """Create a generator function that emits content images & style bottlenecks.

  Args:
    session: tf.Session, the session that contains subgraph to load the traning
      dataset.
    predict_saved_model: str, path to the style predict SavedModel.
    style_dataset: tf.data.Dataset that contains training style images.
    content_dataset: tf.data.Dataset that contains training style images.
    sample_count: int, number of sample to create.

  Returns:
    (str, str) A generator function to use as representative dataset for
    TFLiteConverter.
  """

  # Calculate style bottleneck in advance for representative dataset
  style_bottleneck_list = calculate_style_bottleneck(
      session,
      predict_saved_model,
      style_dataset,
      min_sample_count=sample_count)

  def generator():
    """A generator to be used as representative_dataset for TFLiteConverter."""
    # Get ImageNet data to use as content_image representative dataset
    dataset = content_dataset.batch(1)
    iterator = tf.data.make_initializable_iterator(dataset)
    session.run(iterator.initializer)
    next_element = iterator.get_next()

    # Generate representative dataset
    for index in range(sample_count):
      content_image = session.run(next_element)
      style_bottleneck_input = np.expand_dims(
          style_bottleneck_list[index], axis=0)
      yield [content_image, style_bottleneck_input]

  return generator


def main(unused_argv=None):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Load model weights from trained checkpoint, and export to SavedModel format
  predict_saved_model_folder,  \
    transform_saved_model_folder = export_to_saved_model(
        os.path.expanduser(FLAGS.checkpoint), FLAGS.alpha)

  # Define output TF Lite models path
  output_dir = os.path.expanduser(FLAGS.output_dir)
  if tf.io.gfile.isdir(output_dir):
    tf.logging.warn('Folder %s already existed.' % output_dir)
  else:
    os.mkdir(output_dir)

  # Convert from SavedModel to TensorFlow Lite format
  predict_float_tflite_file = os.path.join(output_dir, 'style_predict.tflite')
  predict_weight_quantized_tflite_file = os.path.join(
      output_dir, 'style_predict_quantized.tflite')
  predict_weight_calibrated_tflite_file = os.path.join(
      output_dir, 'style_predict_calibrated.tflite')
  tranform_float_tflite_file = os.path.join(output_dir,
                                            'style_transform.tflite')
  tranform_weight_quantized_tflite_file = os.path.join(
      output_dir, 'style_transform_quantized.tflite')
  tranform_weight_calibrated_tflite_file = os.path.join(
      output_dir, 'style_transform_calibrated.tflite')

  # Convert the SavedModels to float and weight quantized TF Lite models
  convert_saved_model_to_tflite(
      saved_model_dir=predict_saved_model_folder,
      input_shapes={
          'style_image': [None, FLAGS.image_size, FLAGS.image_size, 3]
      },
      float_tflite_file=predict_float_tflite_file,
      quantized_tflite_file=predict_weight_quantized_tflite_file)

  convert_saved_model_to_tflite(
      saved_model_dir=transform_saved_model_folder,
      input_shapes={
          'content_image': [None, FLAGS.image_size, FLAGS.image_size, 3]
      },
      float_tflite_file=tranform_float_tflite_file,
      quantized_tflite_file=tranform_weight_quantized_tflite_file)

  if (FLAGS.style_dataset_file is not None) and (FLAGS.imagenet_data_dir is
                                                 not None):
    with tf.Session(graph=tf.Graph()) as sess:

      # The training dataset is provided, so we'll do full int8 quantization
      predict_representative_dataset_size = 100
      transform_representative_dataset_size = 300

      style_dataset, content_dataset = get_calibration_dataset(
          image_size=FLAGS.image_size,
          style_dataset_file=FLAGS.style_dataset_file,
          imagenet_data_dir=FLAGS.imagenet_data_dir)

      predict_representative_dataset = predict_model_gen(
          session=sess,
          style_dataset=style_dataset,
          sample_count=predict_representative_dataset_size)

      transform_representative_dataset = transform_model_gen(
          session=sess,
          predict_saved_model=predict_saved_model_folder,
          style_dataset=style_dataset,
          content_dataset=content_dataset,
          sample_count=transform_representative_dataset_size)

      # Convert predict model
      convert_saved_model_to_int8_tflite(
          saved_model_dir=predict_saved_model_folder,
          input_shapes={
              'style_image': [None, FLAGS.image_size, FLAGS.image_size, 3]
          },
          representative_dataset=predict_representative_dataset,
          calibrated_tflite_file=predict_weight_calibrated_tflite_file)

      # Convert transform model
      convert_saved_model_to_int8_tflite(
          saved_model_dir=transform_saved_model_folder,
          input_shapes={
              'content_image': [None, FLAGS.image_size, FLAGS.image_size, 3]
          },
          representative_dataset=transform_representative_dataset,
          calibrated_tflite_file=tranform_weight_calibrated_tflite_file)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
