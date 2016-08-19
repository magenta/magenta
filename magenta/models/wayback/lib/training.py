"""General code for training teacher-forcing sequence models."""
import functools as ft
import numpy as np
import tensorflow as tf

from magenta.models.wayback.lib.namespace import Namespace as NS
import magenta.models.wayback.lib.tfutil as tfutil
import magenta.models.wayback.lib.util as util


class Trainer(object):
  """Sequence model training graph."""

  def __init__(self, model, hp, global_step=None):
    self.model = model
    self.tensors = self._make(hp, global_step=global_step)

  def _make(self, hp, global_step=None):
    """Construct the Tensorflow graph."""
    ts = NS()
    ts.global_step = global_step
    ts.x = tf.placeholder(dtype=tf.int32, name="x")
    length = hp.segment_length + hp.chunk_size
    ts.seq = self.model.make_training_graph(x=ts.x, length=length)
    ts.final_state = ts.seq.final_state
    ts.loss = ts.seq.loss
    ts.error = ts.seq.error

    ts.learning_rate = tf.Variable(hp.initial_learning_rate, dtype=tf.float32,
                                   trainable=False, name="learning_rate")
    ts.decay_op = tf.assign(ts.learning_rate, ts.learning_rate * hp.decay_rate)
    ts.optimizer = tf.train.AdamOptimizer(ts.learning_rate)
    ts.params = tf.trainable_variables()
    print [param.name for param in ts.params]

    ts.gradients = tf.gradients(ts.loss, ts.params)

    loose_params = [
        param for param, gradient in util.equizip(ts.params, ts.gradients)
        if gradient is None]
    if loose_params:
      raise ValueError("loose parameters: %s" %
                       " ".join(param.name for param in loose_params))

    # tensorflow fails miserably to compute gradient for these
    for reg_var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
      ts.gradients[ts.params.index(reg_var)] += (
          hp.weight_decay * tf.gradients(tf.sqrt(tf.reduce_sum(reg_var**2)),
                                         [reg_var])[0])

    ts.clipped_gradients, _ = tf.clip_by_global_norm(ts.gradients, hp.clip_norm)
    ts.training_op = ts.optimizer.apply_gradients(
        util.equizip(ts.clipped_gradients, ts.params),
        global_step=ts.global_step)

    ts.summaries = [
        tf.scalar_summary("loss_train", ts.loss),
        tf.scalar_summary("error_train", ts.error),
        tf.scalar_summary("learning_rate", ts.learning_rate)
    ]
    for parameter, gradient in util.equizip(ts.params,
                                            ts.gradients):
      ts.summaries.append(tf.scalar_summary(
          "meanlogabs_%s" % parameter.name, tfutil.meanlogabs(parameter)))
      ts.summaries.append(tf.scalar_summary(
          "meanlogabsgrad_%s" % parameter.name, tfutil.meanlogabs(gradient)))

    return ts

  def run(self, session, examples, max_step_count=None, hooks=None, hp=None):
    """Train the model using truncated backprop through time.

    Args:
      session: a `tf.Session`.
      examples: a sequence of examples, each of which is represented by a
                sequence of features, each of which is a sequence of data
                points.
      max_step_count: maximum number of SGD steps to take.
      hooks: a Namespace tree containing hook callables. Available hooks:
               step.before: called before taking an SGD step. Receives current
                            training state as argument.
               step.after: called after taking an SGD step. Receives current
                           training state as argument.
      hp: hyperparameters.

    Raises:
      ValueError: if the loss is NaN.
    """
    state = NS(
        global_step=tf.train.global_step(session, self.tensors.global_step),
        model=self.model.initial_state(hp.batch_size))
    while True:
      for batch in util.batches(examples, hp.batch_size):
        for segment in util.segments(batch,
                                     # the last chunk is not processed, so grab
                                     # one more to ensure we backpropagate
                                     # through at least one full model cycle.
                                     # TODO(cotim): rename segment_length to
                                     # backprop_length?
                                     hp.segment_length + hp.chunk_size,
                                     overlap=hp.chunk_size):
          if max_step_count is not None and state.global_step >= max_step_count:
            return

          hooks.Get("step.before", util.noop)(state)
          x, = list(map(util.pad, util.equizip(*segment)))
          feed_dict = {self.tensors.x: x.T}
          feed_dict.update(self.model.feed_dict(state.model))
          values = NS.FlatCall(
              ft.partial(session.run, feed_dict=feed_dict),
              self.tensors.Extract(
                  "loss error summaries global_step training_op learning_rate",
                  "final_state.model"))
          state.model = values.final_state.model
          state.global_step = values.global_step
          hooks.Get("step.after", util.noop)(state, values)

          print ("step #%d loss %f error %f learning rate %e" %
                 (values.global_step, values.loss, values.error,
                  values.learning_rate))

          if np.isnan(values.loss):
            raise ValueError("loss has become NaN")

