# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Generates a stylized image given an unstylized image."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import json

import tensorflow as tf

from tensorflow.python import saved_model

from magenta.models.image_stylization import model, ops


def validate_checkpoint_path(checkpoint):
  if tf.gfile.IsDirectory(checkpoint):
    checkpoint = tf.train.latest_checkpoint(checkpoint)
    tf.logging.info('Using latest checkpoint file at: {}'.format(checkpoint))
  elif not tf.gfile.Exists(checkpoint):
    raise ValueError('No such checkpoint exists: {}'.format(checkpoint))
  return checkpoint


def build_prediction_graph(style_constant, num_styles):
  graph = tf.Graph()
  with graph.as_default():
    image_bytes = tf.placeholder(tf.string, shape=[None])

    images = tf.map_fn(
        functools.partial(tf.image.decode_jpeg, channels=3),
        image_bytes,
        dtype=tf.uint8
    )
    image_floats = tf.cast(images, tf.float32) / 255.0

    if style_constant:
      style_num = tf.constant(
          style_constant,
          shape=[num_styles],
          dtype=tf.float32,
          verify_shape=True
      )
    else:
      # TODO(elibixby) No way to provide an unbatched tensor to Cloud ML Engine
      # Fix to avoid averaging once this capability is added.
      style_weights = tf.placeholder(
          dtype=tf.float32, shape=[None, num_styles])
      style_num = tf.reduce_mean(style_weights, axis=0)

    stylized_images = model.transform(
        image_floats,
        normalizer_fn=ops.weighted_instance_norm,
        normalizer_params={
            'weights': style_num,
            'num_categories': num_styles,
            'center': True,
            # To support abitrarily sized images, no rescaling
            'scale': False
        }
    )
    output_images = tf.cast(stylized_images * 255.0, tf.uint8)

    images = tf.map_fn(tf.image.encode_jpeg, output_images, dtype=tf.string)
    output = tf.encode_base64(images)

    inputs_info = {
        'image_bytes': saved_model.utils.build_tensor_info(image_bytes)
    }
    if not style_constant:
      inputs_info['style_weights'] = saved_model.utils.build_tensor_info(
          style_weights)

    outputs_info = {
        'output_image': saved_model.utils.build_tensor_info(output)
    }

  return graph, inputs_info, outputs_info


def main(checkpoint, output_dir, style_constant, num_styles):
  graph, inputs_info, outputs_info = build_prediction_graph(
      style_constant, num_styles)

  signature_def = saved_model.signature_def_utils.build_signature_def(
      inputs=inputs_info,
      outputs=outputs_info,
      method_name=saved_model.signature_constants.PREDICT_METHOD_NAME)

  checkpoint = validate_checkpoint_path(checkpoint)
  exporter = saved_model.builder.SavedModelBuilder(output_dir)

  with tf.Session(graph=graph) as session:
    saver = tf.train.Saver()
    session.run([tf.local_variables_initializer(), tf.tables_initializer()])
    saver.restore(session, checkpoint)
    exporter.add_meta_graph_and_variables(
        session,
        tags=[saved_model.tag_constants.SERVING],
        signature_def_map={
            saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature_def
        },)

  exporter.save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-styles',
        type=int,
        default=1,
        help='Number of styles the model was trained on.'
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Checkpoint (or checkpoint dir) from which to load the model.'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory. Will be created if it does not exist.'
    )
    parser.add_argument(
        '--style_constant',
        type=json.loads,
        default=None,
        help="""\
            Which styles to use. If unspecified, style is assumed to be an
            input to the SavedModel binary. If specified, this should be a
            list with float values of length [num_styles] which specifies
            the constant weights to be used.\
        """
    )
    args = parser.parse_args()
    main(args.checkpoint, args.output_dir, args.style_constant, args.num_styles)
