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

"""Creates a dataset out of a list of style images.

Each style example in the dataset contains the style image as a JPEG string, a
unique style label and the pre-computed Gram matrices for all layers of a VGG16
classifier pre-trained on Imagenet (where max-pooling operations have been
replaced with average-pooling operations).
"""
import io
import os

from magenta.models.image_stylization import image_utils
from magenta.models.image_stylization import learning
import skimage.io
import tensorflow.compat.v1 as tf

flags = tf.app.flags
flags.DEFINE_string('style_files', None, 'Style image files.')
flags.DEFINE_string('output_file', None, 'Where to save the dataset.')
flags.DEFINE_bool('compute_gram_matrices', True, 'Whether to compute Gram'
                  'matrices or not.')
FLAGS = flags.FLAGS


def _parse_style_files(style_files):
  """Parse the style_files command-line argument."""
  style_files = tf.gfile.Glob(style_files)
  if not style_files:
    raise ValueError('No image files found in {}'.format(style_files))
  return style_files


def _float_feature(value):
  """Creates a float Feature."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  """Creates an int64 Feature."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  """Creates a byte Feature."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  style_files = _parse_style_files(os.path.expanduser(FLAGS.style_files))
  with tf.python_io.TFRecordWriter(
      os.path.expanduser(FLAGS.output_file)) as writer:
    for style_label, style_file in enumerate(style_files):
      tf.logging.info(
          'Processing style file %s: %s' % (style_label, style_file))
      feature = {'label': _int64_feature(style_label)}

      style_image = image_utils.load_np_image(style_file)
      buf = io.BytesIO()
      skimage.io.imsave(buf, style_image, format='JPEG')
      buf.seek(0)
      feature['image_raw'] = _bytes_feature(buf.getvalue())

      if FLAGS.compute_gram_matrices:
        with tf.Graph().as_default():
          style_end_points = learning.precompute_gram_matrices(
              tf.expand_dims(tf.to_float(style_image), 0),
              # We use 'pool5' instead of 'fc8' because a) fully-connected
              # layers are already too deep in the network to be useful for
              # style and b) they're quite expensive to store.
              final_endpoint='pool5')
          for name in style_end_points:
            feature[name] = _float_feature(
                style_end_points[name].flatten().tolist())

      example = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())
  tf.logging.info('Output TFRecord file is saved at %s' % os.path.expanduser(
      FLAGS.output_file))


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
