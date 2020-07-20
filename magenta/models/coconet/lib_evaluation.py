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

# Lint as: python3
"""Helpers for evaluating the log likelihood of pianorolls under a model."""
import time

from magenta.models.coconet import lib_tfutil
from magenta.models.coconet import lib_util
import numpy as np
from scipy.misc import logsumexp
import tensorflow.compat.v1 as tf


def evaluate(evaluator, pianorolls):
  """Evaluate a sequence of pianorolls.

  The returned dictionary contains two kinds of evaluation results: the "unit"
  losses and the "example" losses. The unit loss measures the negative
  log-likelihood of each unit (e.g. note or frame). The example loss is the
  average of the unit loss across the example. Additionally, the dictionary
  contains various aggregates such as the mean and standard error of the mean
  of both losses, as well as min/max and quartile bounds.

  Args:
    evaluator: an instance of BaseEvaluator
    pianorolls: sequence of pianorolls to evaluate

  Returns:
    A dictionary with evaluation results.
  """
  example_losses = []
  unit_losses = []

  for pi, pianoroll in enumerate(pianorolls):
    tf.logging.info("evaluating piece %d", pi)
    start_time = time.time()

    unit_loss = -evaluator(pianoroll)
    example_loss = np.mean(unit_loss)

    example_losses.append(example_loss)
    unit_losses.append(unit_loss)

    duration = (time.time() - start_time) / 60.
    _report(unit_loss, prefix="%i %5.2fmin " % (pi, duration))

    if np.isinf(example_loss):
      break

  _report(example_losses, prefix="FINAL example-level ")
  _report(unit_losses, prefix="FINAL unit-level ")

  rval = dict(example_losses=example_losses, unit_losses=unit_losses)
  rval.update(("example_%s" % k, v) for k, v in _stats(example_losses).items())
  rval.update(
      ("unit_%s" % k, v) for k, v in _stats(_flatcat(unit_losses)).items())
  return rval


def _report(losses, prefix=""):
  tf.logging.info("%s loss %s", prefix, _statstr(_flatcat(losses)))


def _stats(x):
  return dict(
      mean=np.mean(x),
      sem=np.std(x) / np.sqrt(len(x)),
      min=np.min(x),
      max=np.max(x),
      q1=np.percentile(x, 25),
      q2=np.percentile(x, 50),
      q3=np.percentile(x, 75))


def _statstr(x):
  return ("mean/sem: {mean:8.5f}+-{sem:8.5f} {min:.5f} < {q1:.5f} < {q2:.5f} < "
          "{q3:.5f} < {max:.5g}").format(**_stats(x))


def _flatcat(xs):
  return np.concatenate([x.flatten() for x in xs])


class BaseEvaluator(lib_util.Factory):
  """Evaluator base class."""

  def __init__(self, wmodel, chronological):
    """Initialize BaseEvaluator instance.

    Args:
      wmodel: WrappedModel instance
      chronological: whether to evaluate in chronological order or in any order
    """
    self.wmodel = wmodel
    self.chronological = chronological

    def predictor(pianorolls, masks):
      p = self.wmodel.sess.run(
          self.wmodel.model.predictions,
          feed_dict={
              self.wmodel.model.pianorolls: pianorolls,
              self.wmodel.model.masks: masks
          })
      return p

    self.predictor = lib_tfutil.RobustPredictor(predictor)

  @property
  def hparams(self):
    return self.wmodel.hparams

  @property
  def separate_instruments(self):
    return self.wmodel.hparams.separate_instruments

  def __call__(self, pianoroll):
    """Evaluate a single pianoroll.

    Args:
      pianoroll: a single pianoroll, shaped (tt, pp, ii)

    Returns:
      unit losses
    """
    raise NotImplementedError()

  def _update_lls(self, lls, x, pxhat, t, d):
    """Update accumulated log-likelihoods.

    Note: the shape of `lls` and the range of `d` depends on the "number of
    variables per time step" `dd`, which is the number of instruments if
    instruments if instruments are separated or the number of pitches otherwise.

    Args:
      lls: (tt, dd)-shaped array of unit log-likelihoods.
      x: the pianoroll being evaluated, shape (B, tt, P, I).
      pxhat: the probabilities output by the model, shape (B, tt, P, I).
      t: the batch of time indices being evaluated, shape (B,).
      d: the batch of variable indices being evaluated, shape (B,).
    """
    # The code below assumes x is binary, so instead of x * log(px) which is
    # inconveniently NaN if both x and log(px) are zero, we can use
    # where(x, log(px), 0).
    assert np.array_equal(x, x.astype(bool))
    if self.separate_instruments:
      index = (np.arange(x.shape[0]), t, slice(None), d)
    else:
      index = (np.arange(x.shape[0]), t, d, slice(None))
    lls[t, d] = np.log(np.where(x[index], pxhat[index], 1)).sum(axis=1)


class FrameEvaluator(BaseEvaluator):
  """Framewise evaluator.

  Evaluates pianorolls one frame at a time. That is, the model is judged for its
  prediction of entire frames at a time, conditioning on its own samples rather
  than the ground truth of other instruments/pitches in the same frame.

  The frames are evaluated in random order, and within each frame the
  instruments/pitches are evaluated in random order.
  """
  key = "frame"

  def __call__(self, pianoroll):
    tt, pp, ii = pianoroll.shape
    assert self.separate_instruments or ii == 1
    dd = ii if self.separate_instruments else pp

    # Compile a batch with each frame being an example.
    bb = tt
    xs = np.tile(pianoroll[None], [bb, 1, 1, 1])

    ts, ds = self.draw_ordering(tt, dd)

    # Set up sequence of masks to predict the first (according to ordering)
    # instrument for each frame
    mask = []
    mask_scratch = np.ones([tt, pp, ii], dtype=np.float32)
    for j, (t, d) in enumerate(zip(ts, ds)):
      # When time rolls over, reveal the entire current frame for purposes of
      # predicting the next one.
      if j % dd != 0:
        continue
      mask.append(mask_scratch.copy())
      mask_scratch[t, :, :] = 0
    assert np.allclose(mask_scratch, 0)
    del mask_scratch
    mask = np.array(mask)

    lls = np.zeros([tt, dd], dtype=np.float32)

    # We can't parallelize within the frame, as we need the predictions of
    # some of the other instruments.
    # Hence we outer loop over the instruments and parallelize across frames.
    xs_scratch = xs.copy()
    for d_idx in range(dd):
      # Call out to the model to get predictions for the first instrument
      # at each time step.
      pxhats = self.predictor(xs_scratch, mask)

      t, d = ts[d_idx::dd], ds[d_idx::dd]
      assert len(t) == bb and len(d) == bb

      # Write in predictions and update mask.
      if self.separate_instruments:
        xs_scratch[np.arange(bb), t, :, d] = np.eye(pp)[np.argmax(
            pxhats[np.arange(bb), t, :, d], axis=1)]
        mask[np.arange(bb), t, :, d] = 0
        # Every example in the batch sees one frame more than the previous.
        assert np.allclose(
            (1 - mask).sum(axis=(1, 2, 3)),
            [(k * dd + d_idx + 1) * pp for k in range(mask.shape[0])])
      else:
        xs_scratch[np.arange(bb), t, d, :] = (
            pxhats[np.arange(bb), t, d, :] > 0.5)
        mask[np.arange(bb), t, d, :] = 0
        # Every example in the batch sees one frame more than the previous.
        assert np.allclose(
            (1 - mask).sum(axis=(1, 2, 3)),
            [(k * dd + d_idx + 1) * ii for k in range(mask.shape[0])])

      self._update_lls(lls, xs, pxhats, t, d)

    # conjunction over notes within frames; frame is the unit of prediction
    return lls.sum(axis=1)

  def draw_ordering(self, tt, dd):
    o = np.arange(tt, dtype=np.int32)
    if not self.chronological:
      np.random.shuffle(o)
    # random variable orderings within each time step
    o = o[:, None] * dd + np.arange(dd, dtype=np.int32)[None, :]
    for t in range(tt):
      np.random.shuffle(o[t])
    o = o.reshape([tt * dd])
    ts, ds = np.unravel_index(o.T, shape=(tt, dd))
    return ts, ds


class NoteEvaluator(BaseEvaluator):
  """Evalutes note-based negative likelihood."""
  key = "note"

  def __call__(self, pianoroll):
    tt, pp, ii = pianoroll.shape
    assert self.separate_instruments or ii == 1
    dd = ii if self.separate_instruments else pp

    # compile a batch with an example for each variable
    bb = tt * dd
    xs = np.tile(pianoroll[None], [bb, 1, 1, 1])

    ts, ds = self.draw_ordering(tt, dd)
    assert len(ts) == bb and len(ds) == bb

    # set up sequence of masks, one for each variable
    mask = []
    mask_scratch = np.ones([tt, pp, ii], dtype=np.float32)
    for unused_j, (t, d) in enumerate(zip(ts, ds)):
      mask.append(mask_scratch.copy())
      if self.separate_instruments:
        mask_scratch[t, :, d] = 0
      else:
        mask_scratch[t, d, :] = 0
    assert np.allclose(mask_scratch, 0)
    del mask_scratch
    mask = np.array(mask)

    pxhats = self.predictor(xs, mask)

    lls = np.zeros([tt, dd], dtype=np.float32)
    self._update_lls(lls, xs, pxhats, ts, ds)
    return lls

  def _draw_ordering(self, tt, dd):
    o = np.arange(tt * dd, dtype=np.int32)
    if not self.chronological:
      np.random.shuffle(o)
    ts, ds = np.unravel_index(o.T, shape=(tt, dd))
    return ts, ds


class EnsemblingEvaluator(object):
  """Decorating for ensembled evaluation.

  Calls the decorated evaluator multiple times so as to evaluate according to
  multiple orderings. The likelihoods from different orderings are averaged
  in probability space, which gives a better result than averaging in log space
  (which would correspond to a geometric mean that is unnormalized and tends
  to waste probability mass).
  """
  key = "_ensembling"

  def __init__(self, evaluator, ensemble_size):
    self.evaluator = evaluator
    self.ensemble_size = ensemble_size

  def __call__(self, pianoroll):
    lls = [self.evaluator(pianoroll) for _ in range(self.ensemble_size)]
    return logsumexp(lls, b=1. / len(lls), axis=0)
