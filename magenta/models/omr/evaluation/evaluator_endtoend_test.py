"""Runs OMR evaluation end to end and asserts on the evaluation metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

# internal imports
import tensorflow as tf

from magenta.models.omr.evaluation import evaluator
from magenta.models.omr.protobuf import groundtruth_pb2


class EvaluatorEndToEndTest(tf.test.TestCase):

  def testIncludedGroundTruth(self):
    ground_truth = groundtruth_pb2.GroundTruth(
        ground_truth_filename=os.path.join(
            tf.resource_loader.get_data_files_path(),
            '../testdata/IMSLP00747.golden.xml'),
        page_spec=[
            groundtruth_pb2.PageSpec(
                filename=os.path.join(tf.resource_loader.get_data_files_path(),
                                      '../testdata/IMSLP00747-000.png')),
        ])
    results = evaluator.Evaluator().evaluate(ground_truth)
    # Evaluation score is known to be greater than 0.7.
    self.assertGreater(results['overall_score']['total', ''], 0.7)


if __name__ == '__main__':
  tf.test.main()
