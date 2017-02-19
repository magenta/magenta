import numpy as np
import unittest

import magenta.models.wayback.lib.util as util


class UtilTest(unittest.TestCase):

  def setUp(self):
    pass

  def test_equizip(self):
    self.assertRaises(ValueError,
                      lambda: list(util.equizip(range(2), range(3), range(5))))
    self.assertRaises(ValueError,
                      lambda: list(util.equizip(range(5), range(3), range(5))))
    self.assertEqual([(2, 0), (1, 1), (0, 2)],
                     list(util.equizip(reversed(range(3)), range(3))))

  def test_pad(self):
    for _ in range(5):
      m, n = np.random.randint(2, 30, size=[2])
      xs = list(np.random.rand(length, 3, 5)
                for length in np.random.randint(1, n, size=[m]))
      ys = util.pad(xs)
      self.assertEqual(ys.shape[0], m)
      self.assertLessEqual(ys.shape[1], n)
      for i in range(m):
        for j in range(ys.shape[1]):
          # python is the worst
          x = xs[i]
          try:
            x = x[j]
          except IndexError:
            x = 0
          self.assertTrue(np.allclose(x, ys[i][j]))

  def test_segmented_batches(self):
    length = np.random.randint(2, 100)
    segment_length = np.random.randint(1, length)
    example_count = np.random.randint(2, 100)
    batch_size = np.random.randint(1, example_count)
    feature_shapes = [np.random.randint(1, 10, size=np.random.randint(1, 4))
                      for _ in range(np.random.randint(1, 4))]
    examples = [[np.random.rand(length, *shape)
                 for shape in feature_shapes]
                for _ in range(example_count)]

    for batch in util.batches(examples, batch_size, augment=False):
      for segment in util.segments(examples, segment_length):
        self.assertEqual(batch_size, len(batch))
        for features in segment:
          self.assertEqual(len(feature_shapes), len(features))
          for feature, feature_shape in util.equizip(features, feature_shapes):
            self.assertLessEqual(len(feature), segment_length)
            self.assertEqual(tuple(feature_shape), feature.shape[1:])

  def test_segments1(self):
    length = 100
    xs = np.random.randint(2, 100, size=[length])
    examples = [[xs]]
    segment_length = 10

    ys = []
    for segment in util.segments(examples, segment_length):
      ys.extend(segment[0][0])
    self.assertEqual(list(xs), list(ys))

    k = np.random.randint(1, segment_length - 1)
    for i, segment in enumerate(util.segments(examples, segment_length,
                                              overlap=k)):
      if i != 0:
        # pylint: disable=used-before-assignment
        self.assertTrue(np.array_equal(overlap, segment[0][0][:k]))
      overlap = segment[0][0][-k:]

  def test_segments_regression1(self):
    examples = [[np.arange(9)]]
    segment_length = 4
    overlap = 2
    for segment in util.segments(examples, segment_length, overlap=overlap):
      self.assertEqual(4, len(segment[0][0]))

  def test_segments_truncate(self):

    class ComparableNdarray(np.ndarray):
      """A Numpy ndarray that doesn't break equality.

      Numpy ndarray violates the __eq__ contract, which breaks deep
      comparisons. Work around it by wrapping the arrays.
      """

      def __eq__(self, other):
        return np.array_equal(self, other)

    def comparablearray(*args, **kwargs):
      array = np.array(*args, **kwargs)
      return ComparableNdarray(array.shape, buffer=array, dtype=array.dtype)

    def to_examples(segmented_examples):
      segments_by_example = [[[comparablearray(list(s.strip()), dtype="|S1")]
                              for s in e.split("|")]
                             for e in segmented_examples]
      examples_by_segment = list(map(list, util.equizip(*segments_by_example)))
      return examples_by_segment

    examples = to_examples(["abcdefg",
                            "mno",
                            "vwxyz"])[0]

    self.assertEqual(to_examples(["ab|cd|ef|g ",
                                  "mn|o |  |  ",
                                  "vw|xy|z |  "]),
                     list(util.segments(examples, 2, truncate=False)))
    self.assertEqual(to_examples(["ab",
                                  "mn",
                                  "vw"]),
                     list(util.segments(examples, 2, truncate=True)))
    self.assertEqual(to_examples(["abc|def|g  ",
                                  "mno|   |   ",
                                  "vwx|yz |   "]),
                     list(util.segments(examples, 3, truncate=False)))
    self.assertEqual(to_examples(["abc",
                                  "mno",
                                  "vwx"]),
                     list(util.segments(examples, 3, truncate=True)))
    self.assertEqual(to_examples(["abcd|efg ",
                                  "mno |    ",
                                  "vwxy|z   "]),
                     list(util.segments(examples, 4, truncate=False)))
    self.assertEqual([],
                     list(util.segments(examples, 4, truncate=True)))

    overlap = 1
    self.assertEqual(to_examples(["ab|bc|cd|de|ef|fg|g ",
                                  "mn|no|o |  |  |  |  ",
                                  "vw|wx|xy|yz|z |  |  "]),
                     list(util.segments(examples, 2, overlap, truncate=False)))
    self.assertEqual(to_examples(["ab|bc",
                                  "mn|no",
                                  "vw|wx"]),
                     list(util.segments(examples, 2, overlap, truncate=True)))
    self.assertEqual(to_examples(["abc|cde|efg|g  ",
                                  "mno|o  |   |   ",
                                  "vwx|xyz|z  |   "]),
                     list(util.segments(examples, 3, overlap, truncate=False)))
    self.assertEqual(to_examples(["abc",
                                  "mno",
                                  "vwx"]),
                     list(util.segments(examples, 3, overlap, truncate=True)))
    self.assertEqual(to_examples(["abcd|defg|g   ",
                                  "mno |    |    ",
                                  "vwxy|yz  |    "]),
                     list(util.segments(examples, 4, overlap, truncate=False)))
    self.assertEqual([],
                     list(util.segments(examples, 4, overlap, truncate=True)))


if __name__ == "__main__":
  unittest.main()
