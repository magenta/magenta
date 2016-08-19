"""Utility functions.
"""
import collections
import numpy as np
import logging


def odict(*args, **kwargs):
  return collections.OrderedDict(*args, **kwargs)


def constantly(x):
  return lambda: x


def noop(*unused_args, **unused_kwargs):
  pass


def equizip(*iterables):
  """Like `zip` but require that iterables have equal length.

  Args:
    *iterables: The iterables to zip

  Raises:
    ValueError: If the sequences differ in length.

  Yields:
    Tuples like `zip`.
  """
  iterators = list(map(iter, iterables))
  for elements in zip(*iterators):
    yield elements
  # double-check that all iterators are exhausted
  for iterator in iterators:
    try:
      _ = next(iterator)
    except StopIteration:
      pass
    else:
      raise ValueError("Sequences must have equal length")


def augment_by_random_translations(features, num_examples=1):
  """Augment a sequence data point by random circular shifts.

  Args:
    features: list of numpy arrays
      The features that make up the data point.
    num_examples: int
      The number of data points to generate.

  Raises:
    ValueError: If the features differ in length along the first axis.

  Returns:
    The generated data points, as a list of lists of features.
  """
  if not all(feature.shape[0] == features[0].shape[0] for feature in features):
    raise ValueError("Features must have equal length along first axis.")

  length = features[0].shape[0]
  offsets = np.random.choice(length, size=[num_examples], replace=False)
  examples = [[np.roll(feature, -offset, axis=0)
               for feature in features]
              for offset in offsets]
  return examples


def augment_by_slicing(features, num_examples):
  """Augment a sequence data point by slicing it into disjoint examples.

  Args:
    features: list of numpy arrays
      The features that make up the data point.
    num_examples: int
      The number of data points to generate.

  Raises:
    ValueError: If the features differ in length along the first axis.

  Returns:
    The generated data points, as a list of lists of features.
  """
  if not all(feature.shape[0] == features[0].shape[0] for feature in features):
    raise ValueError("Features must have equal length along first axis.")

  length = features[0].shape[0]
  slice_length = length // num_examples
  new_length = num_examples * slice_length

  if new_length < length:
    logging.warning("losing %d elements to augment_by_slicing",
                    length - new_length)

  return list(zip(*[
      feature[:new_length].reshape((num_examples, slice_length))
      for feature in features]))


def batches(examples, batch_size, augment=True):
  """Generate randomly chosen batches of examples.

  If the number of examples is not an integer multiple of `batch_size`, the
  remainder is discarded.

  Args:
    examples: iterable
      The examples to choose from.
    batch_size: int
      The desired number of examples per batch.
    augment: bool
      Whether to augment the examples by random translation.

  Raises:
    ValueError: If `len(examples) < batch_size`.

  Yields:
    Subsets of `batch_size` examples.
  """
  examples = list(examples)

  if len(examples) < batch_size and not augment:
    raise ValueError("Not enough examples to fill a batch.")

  if augment:
    # Generate k derivations for each example to ensure we have at least one
    # batch worth of examples. The originals are discarded; if the augmentation
    # is sensible in the first place then using the originals introduces a bias.
    k = int(np.ceil(batch_size / float(len(examples))))
    examples = [derivation
                for example in examples
                for derivation in augment_by_random_translations(
                    example, num_examples=k)]

  np.random.shuffle(examples)
  for i in range(0, len(examples), batch_size):
    batch = examples[i:i + batch_size]
    if len(batch) < batch_size:
      logging.warning("dropping ragged batch of %d examples", len(batch))
      break
    yield batch


def segments(examples, segment_length, overlap=0, truncate=True):
  """Generate segments from batched sequence data for TBPTT.

  If `truncate` is true, stops as soon as one of the examples runs out, such
  that no padding is needed. Discards the rest.

  Args:
    examples: list of lists of numpy arrays
      The examples to slice up. `examples[i][j]` is the jth feature
      of the ith example.
    segment_length: int
      The desired segment length.
    overlap: int
      The desired number of elements of overlap between consecutive segments.
    truncate: bool
      Whether or not to discard ragged segments where examples vary in length.

  Raises:
    ValueError: If features of an example differ in length or examples differ
    in feature set.

  Yields:
    Slices of examples in the same structure as `examples`. Each segment
    begins where the previous segment left off, except for overlap.
  """
  whichever = 0

  # examples[i][j] is the jth feature of the ith example
  # all features of an example must have the same length:
  if not all(len(examples[i][j]) == len(examples[i][whichever])
             for i, _ in enumerate(examples)
             for j, _ in enumerate(examples[i])):
    raise ValueError("All features of an example must have the same length.")
  # examples[i] and examples[j] must have the same set of features;
  # the same number of features:
  if not all(len(examples[i]) == len(examples[whichever])
             for i, _ in enumerate(examples)):
    raise ValueError("All examples must have the same set of features.")
  # and their shapes must be the same except for length:
  if not all(examples[i][k].shape[1:] == examples[whichever][k].shape[1:]
             for i, _ in enumerate(examples)
             for k, _ in enumerate(examples[i])):
    raise ValueError("All examples must have the same set of features.")

  min_length = min(len(example[whichever]) for example in examples)
  max_length = max(len(example[whichever]) for example in examples)
  max_offset = min_length - segment_length if truncate else max_length - overlap
  for offset in range(0, max_offset + 1, segment_length - overlap):
    # segments[i][j] is a segment of the jth feature of the ith example
    segment = [[feature[offset:offset + segment_length]
                for feature in features]
               for features in examples]
    yield segment


def pad(xs):
  """Zero-pad a list of variable-length numpy arrays.

  The arrays are padded along the first axis and stacked into a single array.

  Args:
    xs: The numpy arrays to pad.

  Returns:
    The resulting array.
  """
  y = np.zeros((len(xs), max(map(len, xs))) + xs[0].shape[1:],
               dtype=xs[0].dtype)
  for i, x in enumerate(xs):
    y[i, :len(x)] = x
  return y


class MeanAggregate(object):
  """Convenience class for cumulative averaging."""

  def __init__(self):
    """Initialize the average.
    """
    self.n = 0
    self.v = 0.

  def add(self, x):
    """Update the average.

    Args:
      x: the new data point to take into account.
    """
    self.v = (self.n * self.v + x) / (self.n + 1)
    self.n += 1

  @property
  def value(self):
    return self.v
