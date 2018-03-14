"""Tests for the staffline patches DoFn graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tempfile

# internal imports
from absl.testing import absltest
import apache_beam as beam
from six import moves
import tensorflow as tf

from magenta.models.omr.training.clustering import staffline_patches_dofn

from tensorflow.python.lib.io import tf_record

PATCH_HEIGHT = 9
PATCH_WIDTH = 7
NUM_STAFFLINES = 9
TIMEOUT_MS = 60000
MAX_PATCHES_PER_PAGE = 10


class StafflinePatchesDoFnTest(absltest.TestCase):

  def testPipeline_corpusImage(self):
    filename = os.path.join(tf.resource_loader.get_data_files_path(),
                            '../../testdata/IMSLP00747-000.png')
    with tempfile.NamedTemporaryFile() as output_examples:
      # Run the pipeline to get the staffline patches.
      with beam.Pipeline() as pipeline:
        dofn = staffline_patches_dofn.StafflinePatchesDoFn(
            PATCH_HEIGHT, PATCH_WIDTH, NUM_STAFFLINES, TIMEOUT_MS,
            MAX_PATCHES_PER_PAGE)
        # pylint: disable=expression-not-assigned
        (pipeline | beam.transforms.Create([filename])
         | beam.transforms.ParDo(dofn) | beam.io.WriteToTFRecord(
             output_examples.name,
             beam.coders.ProtoCoder(tf.train.Example),
             shard_name_template=''))
      # Get the staffline images from a local TensorFlow session.
      with tf.Session() as sess:
        png_path = tf.constant(filename)
        staffline_patches_dofn.pipeline_graph(
            png_path, PATCH_HEIGHT, PATCH_WIDTH, NUM_STAFFLINES)
        stafflines_t = tf.get_default_graph().get_tensor_by_name('stafflines:0')
        stafflines = sess.run(stafflines_t)
      expected_patches = []
      # We should be able to create all possible patches in Python and test the
      # patches from the pipeline.
      for staff in stafflines:
        for staffline in staff:
          for i in moves.range(staffline.shape[1] - PATCH_WIDTH + 1):
            patch = staffline[:, i:i + PATCH_WIDTH]
            expected_patches.append(tuple(patch.ravel()))
      for example_bytes in tf_record.tf_record_iterator(output_examples.name):
        example = tf.train.Example()
        example.ParseFromString(example_bytes)
        patch_pixels = tuple(
            example.features.feature['features'].float_list.value)
        self.assertTrue(patch_pixels in expected_patches)


if __name__ == '__main__':
  absltest.main()
