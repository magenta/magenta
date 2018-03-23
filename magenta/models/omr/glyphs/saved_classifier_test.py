"""Tests running OMR with a dummy saved model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

# internal imports
from backports import tempfile
import numpy as np
import tensorflow as tf

from magenta.models.omr import image
from magenta.models.omr import structure
from magenta.models.omr.glyphs import saved_classifier
from magenta.models.omr.protobuf import musicscore_pb2


class SavedClassifierTest(tf.test.TestCase):

  def testSaveAndLoadDummyClassifier(self):
    with tempfile.TemporaryDirectory() as base_dir:
      export_dir = os.path.join(base_dir, 'export')
      with self.test_session() as sess:
        patches = tf.placeholder(tf.float32, shape=(None, 18, 15))
        num_patches = tf.shape(patches)[0]
        # Glyph.NONE is number 1.
        class_ids = tf.ones([num_patches], tf.int32)
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            # pyformat: disable
            {'input': tf.saved_model.utils.build_tensor_info(patches)},
            {'class_ids': tf.saved_model.utils.build_tensor_info(class_ids)},
            'serve')
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(
            sess, ['serve'],
            signature_def_map={
                tf.saved_model.signature_constants.
                DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    signature
            })
        builder.save()
      tf.reset_default_graph()
      # Load the saved model.
      with self.test_session() as sess:
        filename = os.path.join(tf.resource_loader.get_data_files_path(),
                                '../testdata/IMSLP00747-000.png')
        page = image.decode_music_score_png(tf.read_file(filename))
        clazz = saved_classifier.SavedConvolutional1DClassifier(
            structure.create_structure(page), export_dir)
        predictions = clazz.staffline_predictions.eval()
        self.assertEqual(predictions.ndim, 3)  # Staff, staff position, x
        self.assertGreater(predictions.size, 0)
        # Predictions are all musicscore_pb2.Glyph.NONE.
        self.assertAllEqual(predictions,
                            np.full(predictions.shape,
                                    musicscore_pb2.Glyph.NONE, np.int32))


if __name__ == '__main__':
  tf.test.main()
