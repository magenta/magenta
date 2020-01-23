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

"""Tools for masking out pianorolls in different ways, such as by instrument."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.coconet import lib_util
import numpy as np


class MaskUseError(Exception):
  pass


def apply_mask(pianoroll, mask):
  """Apply mask to pianoroll.

  Args:
    pianoroll: A 3D binary matrix with 2D slices of pianorolls. This is not
        modified.
    mask: A 3D binary matrix with 2D slices of masks, one per each pianoroll.

  Returns:
    A 3D binary matrix with masked pianoroll.

  Raises:
    MaskUseError: If the shape of pianoroll and mask do not match.
  """
  if pianoroll.shape != mask.shape:
    raise MaskUseError("Shape mismatch in pianoroll and mask.")
  return pianoroll * (1 - mask)


def print_mask(mask):
  # assert mask is constant across pitch
  assert np.equal(mask, mask[:, 0, :][:, None, :]).all()
  # get rid of pitch dimension and transpose to get landscape orientation
  mask = mask[:, 0, :].T


def get_mask(maskout_method, *args, **kwargs):
  mm = MaskoutMethod.make(maskout_method)
  return mm(*args, **kwargs)


class MaskoutMethod(lib_util.Factory):
  """Base class for mask distributions used during training."""
  pass


class BernoulliMaskoutMethod(MaskoutMethod):
  """Iid Bernoulli masking distribution."""

  key = "bernoulli"

  def __call__(self,
               pianoroll_shape,
               separate_instruments=True,
               blankout_ratio=0.5,
               **unused_kwargs):
    """Sample a mask.

    Args:
      pianoroll_shape: shape of pianoroll (time, pitch, instrument)
      separate_instruments: whether instruments are separated
      blankout_ratio: bernoulli inclusion probability

    Returns:
      A mask of shape `shape`.

    Raises:
      ValueError: if shape is not three dimensional.
    """
    if len(pianoroll_shape) != 3:
      raise ValueError(
          "Shape needs to of 3 dimensional, time, pitch, and instrument.")
    tt, pp, ii = pianoroll_shape
    if separate_instruments:
      mask = np.random.random([tt, 1, ii]) < blankout_ratio
      mask = mask.astype(np.float32)
      mask = np.tile(mask, [1, pianoroll_shape[1], 1])
    else:
      mask = np.random.random([tt, pp, ii]) < blankout_ratio
      mask = mask.astype(np.float32)
    return mask


class OrderlessMaskoutMethod(MaskoutMethod):
  """Masking distribution for orderless nade training."""

  key = "orderless"

  def __call__(self, shape, separate_instruments=True, **unused_kwargs):
    """Sample a mask.

    Args:
      shape: shape of pianoroll (time, pitch, instrument)
      separate_instruments: whether instruments are separated

    Returns:
      A mask of shape `shape`.
    """
    tt, pp, ii = shape

    if separate_instruments:
      d = tt * ii
    else:
      assert ii == 1
      d = tt * pp
    # sample a mask size
    k = np.random.choice(d) + 1
    # sample a mask of size k
    i = np.random.choice(d, size=k, replace=False)

    mask = np.zeros(d, dtype=np.float32)
    mask[i] = 1.
    if separate_instruments:
      mask = mask.reshape((tt, 1, ii))
      mask = np.tile(mask, [1, pp, 1])
    else:
      mask = mask.reshape((tt, pp, 1))
    return mask
