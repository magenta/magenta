"""General code for evaluating teacher-forcing sequence models."""
import functools as ft
import sys
import numpy as np
import tensorflow as tf

from magenta.models.wayback.lib.namespace import Namespace as NS
import magenta.models.wayback.lib.util as util


class Evaluator(object):
  """Sequence model evaluation graph."""

  def __init__(self, model, hp):
    self.model = model
    self.tensors = self._make(hp)

  def _make(self, unused_hp):
    """Construct the Tensorflow graph."""
    ts = NS()
    ts.x = tf.placeholder(dtype=tf.int32, name="x")
    ts.seq = self.model.make_evaluation_graph(x=ts.x)
    ts.final_state = ts.seq.final_state
    ts.loss = ts.seq.loss
    ts.error = ts.seq.error
    return ts

  def run(self, session, examples, max_step_count=None, hp=None):
    """Evaluate the model.

    Args:
      session: a `tf.Session`.
      examples: a sequence of examples, each of which is represented by a
                sequence of features, each of which is a sequence of data
                points.
      max_step_count: maximum number of segments to process.
      hp: hyperparameters.

    Raises:
      ValueError: if the validation loss is NaN.

    Returns:
      Namespace tree containing statistics and final model state.
    """
    aggregates = NS((key, util.MeanAggregate()) for key in "loss error".split())
    state = NS(step=0, model=self.model.initial_state(hp.batch_size))

    try:
      for batch in util.batches(examples, hp.batch_size):
        for segment in util.segments(batch, hp.segment_length,
                                     overlap=hp.chunk_size):
          if max_step_count is not None and state.step >= max_step_count:
            raise StopIteration()

          x, = list(map(util.pad, util.equizip(*segment)))
          feed_dict = {self.tensors.x: x.T}
          feed_dict.update(self.model.feed_dict(state.model))
          values = NS.FlatCall(
              ft.partial(session.run, feed_dict=feed_dict),
              self.tensors.Extract("loss error final_state.model"))
          state.model = values.final_state.model

          for key in aggregates:
            aggregates[key].add(values[key])

          sys.stderr.write(".")
          state.step += 1
    except StopIteration:
      pass

    sys.stderr.write("\n")

    values = NS((key, aggregate.value) for key, aggregate in aggregates.Items())
    values.summaries = [
        tf.Summary.Value(tag="%s_valid" % key, simple_value=value)
        for key, value in values.Items()]

    print "### evaluation loss %6.5f error %6.5f" % (values.loss, values.error)

    if np.isnan(values.loss):
      raise ValueError("loss has become NaN")

    return values
