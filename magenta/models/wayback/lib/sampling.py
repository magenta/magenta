"""General code for sampling from teacher-forcing sequence models."""
import functools as ft
import sys
import numpy as np
import tensorflow as tf

from magenta.models.wayback.lib.namespace import Namespace as NS
import magenta.models.wayback.lib.util as util


class Sampler(object):
  """Sequence model sampling graph."""

  def __init__(self, model, hp):
    self.model = model
    self.tensors = self._make(hp)

  def _make(self, hp):
    """Construct the Tensorflow graph."""
    ts = NS()
    ts.x = tf.placeholder(dtype=tf.int32, name="x")

    # conditioning graph
    ts.cond = self.model.make_evaluation_graph(x=ts.x)

    # generation graph
    tf.get_variable_scope().reuse_variables()
    ts.initial_xchunk = tf.placeholder(dtype=tf.int32, name="initial_xchunk",
                                       shape=[hp.chunk_size, None])
    ts.length = tf.placeholder(dtype=tf.int32, name="length", shape=[])
    ts.temperature = tf.placeholder(dtype=tf.float32,
                                    name="temperature", shape=[])
    ts.sample = self.model.make_sampling_graph(
        initial_xchunk=ts.initial_xchunk, length=ts.length,
        temperature=ts.temperature)

    return ts

  def run(self, session, primers, length, temperature, hp=None):
    """Sample from the model.

    Args:
      session: a `tf.Session`.
      primers: a sequence of examples, each of which is represented by a
               sequence of features, each of which is a sequence of data
               points. Used to condition the model before sampling.
      length: the desired length of the sample.
      temperature: softmax temperature.
      hp: hyperparameters.

    Returns:
      the sampled sequences, in a numpy array of shape `[len(x), length]`.
    """
    batch_size = len(primers)
    # process in segments to avoid tensorflow eating all the memory
    max_segment_length = min(10000, hp.segment_length)

    print "conditioning..."
    segment_length = min(max_segment_length,
                         max(len(primer[0]) for primer in primers))
    # ensure segment_length is a multiple of chunk_size
    segment_length -= segment_length % hp.chunk_size

    state = NS(model=self.model.initial_state(batch_size))
    for segment in util.segments(
        primers, segment_length, overlap=hp.chunk_size):
      x, = list(map(util.pad, util.equizip(*segment)))
      feed_dict = {self.tensors.x: x.T}
      feed_dict.update(self.model.feed_dict(state.model))
      values = NS.FlatCall(
          ft.partial(session.run, feed_dict=feed_dict),
          self.tensors.cond.Extract("final_state.model final_xchunk"))
      state.model = values.final_state.model
      sys.stderr.write(".")
    sys.stderr.write("\n")

    cond_values = values

    # make sure length is a multiple of chunk_size
    chunky_length = length + hp.chunk_size - length % hp.chunk_size

    print "sampling..."
    length_left = chunky_length
    xhats = []
    state = NS(model=cond_values.final_state.model,
               initial_xchunk=cond_values.final_xchunk)
    while length_left > 0:
      segment_length = min(max_segment_length, length_left)
      length_left -= segment_length

      feed_dict = {self.tensors.initial_xchunk: state.initial_xchunk,
                   self.tensors.length: segment_length,
                   self.tensors.temperature: temperature}
      feed_dict.update(self.model.feed_dict(state.model))
      sample_values = NS.FlatCall(
          ft.partial(session.run, feed_dict=feed_dict),
          self.tensors.sample.Extract("final_state.model xhat final_xhatchunk"))
      state.model = sample_values.final_state.model
      state.initial_xchunk = sample_values.final_xhatchunk

      xhats.append(sample_values.xhat)
      sys.stderr.write(".")
    sys.stderr.write("\n")

    xhat = np.concatenate(xhats, axis=0)
    # truncate from chunky_length to the desired sample length
    xhat = xhat[:length]
    return xhat.T
