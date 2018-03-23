"""Tests for run length encoding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import tensorflow as tf

from magenta.models.omr.util import run_length


class RunLengthTest(tf.test.TestCase):

  def testEmpty(self):
    with self.test_session() as sess:
      columns, values, lengths = sess.run(
          run_length.vertical_run_length_encoding(tf.zeros((0, 0), tf.bool)))
    self.assertAllEqual(columns, [])
    self.assertAllEqual(values, [])
    self.assertAllEqual(lengths, [])

  def testBooleanImage(self):
    img = tf.cast(
        [
            [0, 0, 1, 0, 0, 1],
            # pyformat: disable
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 1, 0]
        ],
        tf.bool)
    with self.test_session() as sess:
      columns, values, lengths = sess.run(
          run_length.vertical_run_length_encoding(img))
    self.assertAllEqual(columns,
                        [0] * 3 + [1] * 4 + [2] + [3] * 3 + [4] * 2 + [5] * 2)
    self.assertAllEqual(values, [0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    self.assertAllEqual(lengths, [1, 1, 2, 1, 1, 1, 1, 4, 1, 2, 1, 1, 3, 3, 1])


if __name__ == '__main__':
  tf.test.main()
