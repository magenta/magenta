"""Tests for the functional ops helpers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import tensorflow as tf

from magenta.models.omr.util import functional_ops


class FunctionalOpsTest(tf.test.TestCase):

  def testFlatMap(self):
    with self.test_session():
      items = functional_ops.flat_map_fn(tf.range, [1, 3, 0, 5])
      self.assertAllEqual(items.eval(), [0, 0, 1, 2, 0, 1, 2, 3, 4])


if __name__ == '__main__':
  tf.test.main()
