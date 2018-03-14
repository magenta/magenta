"""Provides the default included OMR classifier.

This assumes a `saved_model.pb` file with no variables in the model (which would
be separate files).

This is an internal version that can be run in a .PAR file. This module will not
be shared between Piper and Git, which will have a simpler open source version.
The open source version is just a wrapper around the
`SavedConvolutional1DClassifier` ctor, which assumes that the saved model is a
real directory.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

# internal imports
import tensorflow as tf

from magenta.models.omr.glyphs import saved_classifier

_SAVED_MODEL_PATH = '../data/knn_saved_model_20170514'


def build_classifier_fn(saved_model=None):
  """Returns a glyph classifier fn for a saved model.

  The result can be given to `OMREngine` to configure the saved model to use.

  Args:
    saved_model: Saved model directory. If None, uses the default KNN saved
        model included with Magenta.

  Returns:
    A callable that accepts a `Structure` and returns a `BaseGlyphClassifier`.
  """
  saved_model = (
      saved_model or
      os.path.join(tf.resource_loader.get_data_files_path(), _SAVED_MODEL_PATH))
  ctor = saved_classifier.SavedConvolutional1DClassifier
  return lambda structure: ctor(structure, saved_model)
