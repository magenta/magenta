"""Tests for staffline patches k-means."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

# internal imports
from absl import flags
from absl.testing import absltest
import numpy as np

from magenta.models.omr.training.clustering import staffline_patches_kmeans_pipeline

from tensorflow.core.example import example_pb2
from tensorflow.python.lib.io import tf_record

FLAGS = flags.FLAGS

NUM_CLUSTERS = 20
BATCH_SIZE = 100
TRAIN_STEPS = 5


class StafflinePatchesKmeansPipelineTest(absltest.TestCase):

  def testKmeans(self):
    num_features = FLAGS.patch_height * FLAGS.patch_width
    dummy_data = np.random.random((500, num_features))
    with tempfile.NamedTemporaryFile(mode='r') as patches_file:
      with tf_record.TFRecordWriter(patches_file.name) as patches_writer:
        for patch in dummy_data:
          example = example_pb2.Example()
          example.features.feature['features'].float_list.value.extend(patch)
          patches_writer.write(example.SerializeToString())
      clusters = staffline_patches_kmeans_pipeline.train_kmeans(
          patches_file.name, NUM_CLUSTERS, BATCH_SIZE, TRAIN_STEPS)
      self.assertEqual(clusters.shape, (NUM_CLUSTERS, num_features))


if __name__ == '__main__':
  absltest.main()
