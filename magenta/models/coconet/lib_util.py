# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for context managing, data prep and sampling such as softmax."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import datetime
import numbers
import pdb
import tempfile
import time

import numpy as np
import tensorflow.compat.v1 as tf


@contextlib.contextmanager
def atomic_file(path):
  """Atomically saves data to a target path.

  Any existing data at the target path will be overwritten.

  Args:
    path: target path at which to save file

  Yields:
    file-like object
  """
  with tempfile.NamedTemporaryFile() as tmp:
    yield tmp
    tmp.flush()
    tf.gfile.Copy(tmp.name, "%s.tmp" % path, overwrite=True)
  tf.gfile.Rename("%s.tmp" % path, path, overwrite=True)


def sample_bernoulli(p, temperature=1):
  """Sample an array of Bernoullis.

  Args:
    p: an array of Bernoulli probabilities.
    temperature: if not 1, transform the distribution by dividing the log
        probabilities and renormalizing. Values greater than 1 increase entropy,
        values less than 1 decrease entropy. A value of 0 yields a deterministic
        distribution that chooses the mode.

  Returns:
    A binary array of sampled values, the same shape as `p`.
  """
  if temperature == 0.:
    sampled = p > 0.5
  else:
    pp = np.stack([p, 1 - p])
    logpp = np.log(pp)
    logpp /= temperature
    logpp -= logpp.max(axis=0, keepdims=True)
    p = np.exp(logpp)
    p /= p.sum(axis=0)
    print("%.5f < %.5f < %.5f < %.5f < %.5g" % (np.min(p), np.percentile(p, 25),
                                                np.percentile(p, 50),
                                                np.percentile(p, 75),
                                                np.max(p)))

    sampled = np.random.random(p.shape) < p
  return sampled


def softmax(p, axis=None, temperature=1):
  """Apply the softmax transform to an array of categorical distributions.

  Args:
    p: an array of categorical probability vectors, possibly unnormalized.
    axis: the axis that spans the categories (default: -1).
    temperature: if not 1, transform the distribution by dividing the log
        probabilities and renormalizing. Values greater than 1 increase entropy,
        values less than 1 decrease entropy. A value of 0 yields a deterministic
        distribution that chooses the mode.

  Returns:
    An array of categorical probability vectors, like `p` but tempered and
    normalized.
  """
  if axis is None:
    axis = p.ndim - 1
  if temperature == 0.:
    # NOTE: in case of multiple equal maxima, returns uniform distribution.
    p = p == np.max(p, axis=axis, keepdims=True)
  else:
    # oldp = p
    logp = np.log(p)
    logp /= temperature
    logp -= logp.max(axis=axis, keepdims=True)
    p = np.exp(logp)
  p /= p.sum(axis=axis, keepdims=True)
  if np.isnan(p).any():
    pdb.set_trace()
  return p


def sample(p, axis=None, temperature=1, onehot=False):
  """Sample an array of categorical variables.

  Args:
    p: an array of categorical probability vectors, possibly unnormalized.
    axis: the axis that spans the categories (default: -1).
    temperature: if not 1, transform the distribution by dividing the log
        probabilities and renormalizing. Values greater than 1 increase entropy,
        values less than 1 decrease entropy. A value of 0 yields a deterministic
        distribution that chooses the mode.
    onehot: whether to one-hot encode the result.

  Returns:
    An array of samples. If `onehot` is False, the result is an array of integer
    category indices, with the categorical axis removed. If `onehot` is True,
    these indices are one-hot encoded, so that the categorical axis remains and
    the result has the same shape and dtype as `p`.
  """
  assert (p >=
          0).all()  # just making sure we don't put log probabilities in here

  if axis is None:
    axis = p.ndim - 1

  if temperature != 1:
    p **= (1. / temperature)
  cmf = p.cumsum(axis=axis)
  totalmasses = cmf[tuple(
      slice(None) if d != axis else slice(-1, None) for d in range(cmf.ndim))]
  u = np.random.random([p.shape[d] if d != axis else 1 for d in range(p.ndim)])
  i = np.argmax(u * totalmasses < cmf, axis=axis)

  return to_onehot(i, axis=axis, depth=p.shape[axis]) if onehot else i


def to_onehot(i, depth, axis=-1):
  """Convert integer categorical indices to one-hot probability vectors.

  Args:
    i: an array of integer categorical indices.
    depth: the number of categories.
    axis: the axis on which to lay out the categories.

  Returns:
    An array of one-hot categorical indices, shaped like `i` but with a
    categorical axis in the location specified by `axis`.
  """
  x = np.eye(depth)[i]
  axis %= x.ndim
  if axis != x.ndim - 1:
    # move new axis forward
    axes = list(range(x.ndim - 1))
    axes.insert(axis, x.ndim - 1)
    x = np.transpose(x, axes)
  assert np.allclose(x.sum(axis=axis), 1)
  return x


def deepsubclasses(klass):
  """Iterate over direct and indirect subclasses of `klass`."""
  for subklass in klass.__subclasses__():
    yield subklass
    for subsubklass in deepsubclasses(subklass):
      yield subsubklass


class Factory(object):
  """Factory mixin.

  Provides a `make` method that searches for an appropriate subclass to
  instantiate given a key. Subclasses inheriting from a class that has Factory
  mixed in can expose themselves for instantiation through this method by
  setting the class attribute named `key` to an appropriate value.
  """

  @classmethod
  def make(cls, key, *args, **kwargs):
    """Instantiate a subclass of `cls`.

    Args:
      key: the key identifying the subclass.
      *args: passed on to the subclass constructor.
      **kwargs: passed on to the subclass constructor.

    Returns:
      An instantiation of the subclass that has the given key.

    Raises:
      KeyError: if key is not a child subclass of cls.
    """
    for subklass in deepsubclasses(cls):
      if subklass.key == key:
        return subklass(*args, **kwargs)

    raise KeyError("unknown %s subclass key %s" % (cls, key))


@contextlib.contextmanager
def timing(label, printon=True):
  """Context manager that times and logs execution."""
  if printon:
    print("enter %s" % label)
  start_time = time.time()
  yield
  time_taken = (time.time() - start_time) / 60.0
  if printon:
    print("exit  %s (%.3fmin)" % (label, time_taken))


class AggregateMean(object):
  """Aggregates values for mean."""

  def __init__(self, name):
    self.name = name
    self.value = 0.
    self.total_counts = 0

  def add(self, value, counts=1):
    """Add an amount to the total and also increment the counts."""
    self.value += value
    self.total_counts += counts

  @property
  def mean(self):
    """Return the mean."""
    return self.value / self.total_counts


def timestamp():
  return datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def get_rng(rng=None):
  if rng is None:
    return np.random
  if isinstance(rng, numbers.Integral):
    return np.random.RandomState(rng)
  else:
    return rng


@contextlib.contextmanager
def numpy_seed(seed):
  """Context manager that temporarily sets the numpy.random seed."""
  if seed is not None:
    prev_rng_state = np.random.get_state()
    np.random.seed(seed)
  yield
  if seed is not None:
    np.random.set_state(prev_rng_state)


def random_crop(x, length):
  leeway = len(x) - length
  start = np.random.randint(1 + max(0, leeway))
  x = x[start:start + length]
  return x


def batches(*xss, **kwargs):
  """Iterate over subsets of lists of examples.

  Yields batches of the form `[xs[indices] for xs in xss]` where at each
  iteration `indices` selects a subset. Each index is only selected once.
  **kwards could be one of the following:
    size: number of elements per batch
    discard_remainder: if true, discard final short batch
    shuffle: if true, yield examples in randomly determined order
    shuffle_rng: seed or rng to determine random order

  Args:
    *xss: lists of elements to batch
    **kwargs: kwargs could be one of the above.


  Yields:
    A batch of the same structure as `xss`, but with `size` examples.
  """
  size = kwargs.get("size", 1)
  discard_remainder = kwargs.get("discard_remainder", True)
  shuffle = kwargs.get("shuffle", False)
  shuffle_rng = kwargs.get("shuffle_rng", None)

  shuffle_rng = get_rng(shuffle_rng)
  n = int(np.unique(list(map(len, xss))))
  assert all(len(xs) == len(xss[0]) for xs in xss)
  indices = np.arange(len(xss[0]))
  if shuffle:
    np.random.shuffle(indices)
  for start in range(0, n, size):
    batch_indices = indices[start:start + size]
    if len(batch_indices) < size and discard_remainder:
      break
    batch_xss = [xs[batch_indices] for xs in xss]
    yield batch_xss


def pad_and_stack(*xss):
  """Pad and stack lists of examples.

  Each argument `xss[i]` is taken to be a list of variable-length examples.
  The examples are padded to a common length and stacked into an array.
  Example lengths must match across the `xss[i]`.

  Args:
    *xss: lists of examples to stack

  Returns:
    A tuple `(yss, lengths)`. `yss` is a list of arrays of padded examples,
    each `yss[i]` corresponding to `xss[i]`. `lengths` is an array of example
    lengths.
  """
  yss = []
  lengths = list(map(len, xss[0]))
  for xs in xss:
    # example lengths must be the same across arguments
    assert lengths == list(map(len, xs))
    max_length = max(lengths)
    rest_shape = xs[0].shape[1:]
    ys = np.zeros((len(xs), max_length) + rest_shape, dtype=xs[0].dtype)
    for i in range(len(xs)):
      ys[i, :len(xs[i])] = xs[i]
    yss.append(ys)
  return list(map(np.asarray, yss)), np.asarray(lengths)


def identity(x):
  return x


def eqzip(*xss):
  """Zip iterables of the same length.

  Unlike the builtin `zip`, this fails on iterables of different lengths.
  As a side-effect, it exhausts (and stores the elements of) all iterables
  before starting iteration.

  Args:
    *xss: the iterables to zip.

  Returns:
    zip(*xss)

  Raises:
    ValueError: if the iterables are of different lengths.
  """
  xss = list(map(list, xss))
  lengths = list(map(len, xss))
  if not all(length == lengths[0] for length in lengths):
    raise ValueError("eqzip got iterables of unequal lengths %s" % lengths)
  return zip(*xss)
