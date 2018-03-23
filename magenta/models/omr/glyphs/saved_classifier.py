"""Saved patch classifier models for OMR.

The saved model should accept a 3D tensor of patches
`(num_patches, patch_height, patch_width)`, and return a tensor of shape
`(num_patches, len(musicscore_pb2.Glyph.Type.keys()))`. Patches are extracted
from each vertical position on each staff where we expect to find glyphs, and
any arbitrary model can be loaded here.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import tensorflow as tf

from magenta.models.omr.glyphs import convolutional
from magenta.models.omr.staves import staffline_extractor
from magenta.models.omr.util import patches

from tensorflow.python.estimator.canned import prediction_keys


class SavedConvolutional1DClassifier(
    convolutional.Convolutional1DGlyphClassifier):
  """Holds a saved glyph classifier model.

  To use a saved glyph classifier with `OMREngine`, see the
  `saved_classifier_fn` wrapper.
  """

  def __init__(self,
               structure,
               saved_model_dir,
               num_sections=19,
               *args,
               **kwargs):
    """Loads a saved classifier model for the OMR engine.

    Args:
      structure: A `structure.Structure`.
      saved_model_dir: Path to the TF saved_model directory to load.
      num_sections: Number of vertical positions of patches to extract, centered
        on the middle staff line.
      *args: Passed through to `SavedConvolutional1DClassifier`.
      **kwargs: Passed through to `SavedConvolutional1DClassifier`.

    Raises:
      ValueError: If the saved model input could not be interpreted as a 3D
        array with the patch size.
    """
    super(SavedConvolutional1DClassifier, self).__init__(*args, **kwargs)
    sess = tf.get_default_session()
    graph_def = tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
    signature = graph_def.signature_def[
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    input_info = signature.inputs['input']
    if not (len(input_info.tensor_shape.dim) == 3 and
            input_info.tensor_shape.dim[1].size > 0 and
            input_info.tensor_shape.dim[2].size > 0):
      raise ValueError('Invalid patches input: ' + str(input_info))
    patch_height = input_info.tensor_shape.dim[1].size
    patch_width = input_info.tensor_shape.dim[2].size

    with tf.name_scope('saved_classifier'):
      self.staffline_extractor = staffline_extractor.StafflineExtractor(
          structure.staff_remover.remove_staves,
          structure.staff_detector,
          num_sections=num_sections,
          target_height=patch_height)
      stafflines = self.staffline_extractor.extract_staves()
      num_staves = tf.shape(stafflines)[0]
      num_sections = tf.shape(stafflines)[1]
      staffline_patches = patches.patches_1d(stafflines, patch_width)
      staffline_patches_shape = tf.shape(staffline_patches)
      patches_per_position = staffline_patches_shape[2]
      flat_patches = tf.reshape(staffline_patches, [
          num_staves * num_sections * patches_per_position, patch_height,
          patch_width
      ])

      # Feed in the flat extracted patches as the classifier input.
      predictions_name = signature.outputs[
          prediction_keys.PredictionKeys.CLASS_IDS].name
      predictions = tf.contrib.graph_editor.graph_replace(
          sess.graph.get_tensor_by_name(predictions_name), {
              sess.graph.get_tensor_by_name(signature.inputs['input'].name):
                  flat_patches
          })
      # Reshape to the original patches shape.
      predictions = tf.reshape(predictions, staffline_patches_shape[:3])

      # Pad the output. We take only the valid patches, but we want to shift all
      # of the predictions so that a patch at index i on the x-axis is centered
      # on column i. This determines the x coordinates of the glyphs.
      width = tf.shape(stafflines)[-1]
      predictions_width = tf.shape(predictions)[-1]
      pad_before = (width - predictions_width) // 2
      pad_shape_before = tf.concat(
          [staffline_patches_shape[:2], [pad_before]], axis=0)
      pad_shape_after = tf.concat(
          [
              staffline_patches_shape[:2],
              [width - predictions_width - pad_before]
          ],
          axis=0)
      self.output = tf.concat(
          [
              # NONE has value 1.
              tf.ones(pad_shape_before, tf.int64),
              tf.to_int64(predictions),
              tf.ones(pad_shape_after, tf.int64),
          ],
          axis=-1)

  @property
  def staffline_predictions(self):
    return self.output
