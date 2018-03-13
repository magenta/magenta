"""Utility for exporting Coconet to SavedModel."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# internal imports
import tensorflow as tf

from magenta.models.coconet import lib_graph


FLAGS = tf.app.flags.FLAGS
flags = tf.app.flags
flags.DEFINE_string('checkpoint', None,
                    'Path to the checkpoint to export.')
flags.DEFINE_string('destination', None,
                    'Path to export SavedModel.')


def export_saved_model(model, destination):
  """Exports the given model as SavedModel to destination."""
  if model is None or destination is None or not destination:
    tf.logging.error('No model or destination provided.')
    return

  builder = tf.saved_model.builder.SavedModelBuilder(destination)

  signature = tf.saved_model.signature_def_utils.predict_signature_def(
      inputs={
          'pianorolls': model.model.pianorolls,
          'masks': model.model.masks,
          'lengths': model.model.lengths,
      }, outputs={
          'predictions': model.model.predictions
      })

  signature_def_map = {
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
      signature}

  builder.add_meta_graph_and_variables(
      model.sess,
      [tf.saved_model.tag_constants.SERVING],
      signature_def_map=signature_def_map)
  builder.save()


def load_saved_model(sess, path):
  """Loads the SavedModel at path."""
  tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], path)


def main(unused_argv):
  if FLAGS.checkpoint is None or not FLAGS.checkpoint:
    raise ValueError(
        'Need to provide a path to checkpoint directory.')
  if FLAGS.destination is None or not FLAGS.destination:
    raise ValueError(
        'Need to provide a destination directory for the SavedModel.')
  model = lib_graph.load_checkpoint(FLAGS.checkpoint)
  export_saved_model(model, FLAGS.destination)
  tf.logging.info('Exported SavedModel to %s.', FLAGS.destination)


if __name__ == '__main__':
  tf.app.run()
