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

"""Utility for exporting and loading a Coconet SavedModel."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def get_signature_def(model, use_tf_sampling):
  """Creates a signature def for the SavedModel."""
  if use_tf_sampling:
    return tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={
            'pianorolls': model.inputs['pianorolls'],
        }, outputs={
            'predictions': tf.cast(model.samples, tf.bool),
        })
  return tf.saved_model.signature_def_utils.predict_signature_def(
      inputs={
          'pianorolls': model.model.pianorolls,
          'masks': model.model.masks,
          'lengths': model.model.lengths,
      }, outputs={
          'predictions': model.model.predictions
      })


def export_saved_model(model, destination, tags, use_tf_sampling):
  """Exports the given model as SavedModel to destination."""
  if model is None or destination is None or not destination:
    tf.logging.error('No model or destination provided.')
    return

  builder = tf.saved_model.builder.SavedModelBuilder(destination)

  signature_def_map = {
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
      get_signature_def(model, use_tf_sampling)}

  builder.add_meta_graph_and_variables(
      model.sess,
      tags,
      signature_def_map=signature_def_map,
      strip_default_attrs=True)
  builder.save()


def load_saved_model(sess, path, tags):
  """Loads the SavedModel at path."""
  tf.saved_model.loader.load(sess, tags, path)
