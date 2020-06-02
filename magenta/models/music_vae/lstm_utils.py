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

"""MusicVAE LSTM model utilities."""
import collections

from magenta.contrib import rnn as contrib_rnn
from magenta.contrib import seq2seq as contrib_seq2seq
import tensorflow.compat.v1 as tf

rnn = tf.nn.rnn_cell


def rnn_cell(rnn_cell_size, dropout_keep_prob, residual, is_training=True):
  """Builds an LSTMBlockCell based on the given parameters."""
  dropout_keep_prob = dropout_keep_prob if is_training else 1.0
  cells = []
  for i in range(len(rnn_cell_size)):
    cell = contrib_rnn.LSTMBlockCell(rnn_cell_size[i])
    if residual:
      cell = rnn.ResidualWrapper(cell)
      if i == 0 or rnn_cell_size[i] != rnn_cell_size[i - 1]:
        cell = contrib_rnn.InputProjectionWrapper(cell, rnn_cell_size[i])
    cell = rnn.DropoutWrapper(
        cell,
        input_keep_prob=dropout_keep_prob)
    cells.append(cell)
  return rnn.MultiRNNCell(cells)


def build_bidirectional_lstm(
    layer_sizes, dropout_keep_prob, residual, is_training):
  """Build the Tensorflow graph for a bidirectional LSTM."""

  cells_fw = []
  cells_bw = []
  for layer_size in layer_sizes:
    cells_fw.append(
        rnn_cell([layer_size], dropout_keep_prob, residual, is_training))
    cells_bw.append(
        rnn_cell([layer_size], dropout_keep_prob, residual, is_training))

  return cells_fw, cells_bw


def _get_final_index(sequence_length, time_major=True):
  indices = [tf.maximum(0, sequence_length - 1),
             tf.range(sequence_length.shape[0])]
  if not time_major:
    indices = indices[-1::-1]
  return tf.stack(indices, axis=1)


def get_final(sequence, sequence_length, time_major=True):
  """Get the final item in a batch of sequences."""
  final_index = _get_final_index(sequence_length, time_major)
  return tf.gather_nd(sequence, final_index)


def set_final(sequence, sequence_length, values, time_major=False):
  """Sets the final values in a batch of sequences, and clears those after."""
  sequence_batch_major = (
      sequence if not time_major else tf.transpose(sequence, [1, 0, 2]))
  final_index = _get_final_index(sequence_length, time_major=False)
  mask = tf.sequence_mask(
      tf.maximum(0, sequence_length - 1),
      maxlen=sequence_batch_major.shape[1],
      dtype=tf.float32)
  sequence_batch_major = (
      tf.expand_dims(mask, axis=-1) * sequence_batch_major +
      tf.scatter_nd(final_index, values, tf.shape(sequence_batch_major)))
  if time_major:
    return tf.transpose(sequence_batch_major, [1, 0, 2])
  return sequence_batch_major


def initial_cell_state_from_embedding(cell, z, name=None):
  """Computes an initial RNN `cell` state from an embedding, `z`."""
  flat_state_sizes = tf.nest.flatten(cell.state_size)
  return tf.nest.pack_sequence_as(
      cell.zero_state(batch_size=z.shape[0], dtype=tf.float32),
      tf.split(
          tf.layers.dense(
              z,
              sum(flat_state_sizes),
              activation=tf.tanh,
              kernel_initializer=tf.random_normal_initializer(stddev=0.001),
              name=name),
          flat_state_sizes,
          axis=1))


def get_sampling_probability(hparams, is_training):
  """Returns the sampling probability as a tensor based on the hparams.

  Supports three sampling schedules (`hparams.sampling_schedule`):
    constant: `hparams.sampling_rate` is the sampling probability. Must be in
      the interval [0, 1].
    exponential: `hparams.sampling_rate` is the base of the decay exponential.
      Must be in the interval (0, 1). Larger values imply a slower increase in
      sampling.
    inverse_sigmoid: `hparams.sampling_rate` is in the interval [1, inf).
      Larger values imply a slower increase in sampling.

  A constant value of 0 is returned if `hparams.sampling_schedule` is undefined.

  If not training and a non-0 sampling schedule is defined, a constant value of
  1 is returned since this is assumed to be a test/eval job associated with a
  scheduled sampling trainer.

  Args:
    hparams: An HParams object containing model hyperparameters.
    is_training: Whether or not the model is being used for training.

  Raises:
    ValueError: On an invalid `sampling_schedule` or `sampling_rate` hparam.
  """
  if (not hasattr(hparams, 'sampling_schedule') or
      not hparams.sampling_schedule or
      (hparams.sampling_schedule == 'constant' and hparams.sampling_rate == 0)):
    return tf.constant(0.0)

  if not is_training:
    # This is likely an eval/test job associated with a training job using
    # scheduled sampling.
    tf.logging.warning(
        'Setting non-training sampling schedule from %s:%f to constant:1.0.',
        hparams.sampling_schedule, hparams.sampling_rate)
    hparams.sampling_schedule = 'constant'
    hparams.sampling_rate = 1.0

  schedule = hparams.sampling_schedule
  rate = hparams.sampling_rate
  step = tf.to_float(tf.train.get_global_step())

  if schedule == 'constant':
    if not 0 <= rate <= 1:
      raise ValueError(
          '`constant` sampling rate must be in the interval [0, 1]. Got %f.'
          % rate)
    sampling_probability = tf.to_float(rate)
  elif schedule == 'inverse_sigmoid':
    if rate < 1:
      raise ValueError(
          '`inverse_sigmoid` sampling rate must be at least 1. Got %f.' % rate)
    k = tf.to_float(rate)
    sampling_probability = 1.0 - k / (k + tf.exp(step / k))
  elif schedule == 'exponential':
    if not 0 < rate < 1:
      raise ValueError(
          '`exponential` sampling rate must be in the interval (0, 1). Got %f.'
          % hparams.sampling_rate)
    k = tf.to_float(rate)
    sampling_probability = 1.0 - tf.pow(k, step)
  else:
    raise ValueError('Invalid `sampling_schedule`: %s' % schedule)
  tf.summary.scalar('sampling_probability', sampling_probability)
  return sampling_probability


class LstmDecodeResults(
    collections.namedtuple('LstmDecodeResults',
                           ('rnn_input', 'rnn_output', 'samples', 'final_state',
                            'final_sequence_lengths'))):
  pass


class Seq2SeqLstmDecoderOutput(
    collections.namedtuple('BasicDecoderOutput',
                           ('rnn_input', 'rnn_output', 'sample_id'))):
  pass


class Seq2SeqLstmDecoder(contrib_seq2seq.BasicDecoder):
  """Overrides BaseDecoder to include rnn inputs in the output."""

  def __init__(self, cell, helper, initial_state, input_shape,
               output_layer=None):
    self._input_shape = input_shape
    super(Seq2SeqLstmDecoder, self).__init__(
        cell, helper, initial_state, output_layer)

  @property
  def output_size(self):
    return Seq2SeqLstmDecoderOutput(
        rnn_input=self._input_shape,
        rnn_output=self._rnn_output_size(),
        sample_id=self._helper.sample_ids_shape)

  @property
  def output_dtype(self):
    dtype = tf.nest.flatten(self._initial_state)[0].dtype
    return Seq2SeqLstmDecoderOutput(
        dtype,
        tf.nest.map_structure(lambda _: dtype, self._rnn_output_size()),
        self._helper.sample_ids_dtype)

  def step(self, time, inputs, state, name=None):
    results = super(Seq2SeqLstmDecoder, self).step(time, inputs, state, name)
    outputs = Seq2SeqLstmDecoderOutput(
        rnn_input=inputs,
        rnn_output=results[0].rnn_output,
        sample_id=results[0].sample_id)
    return (outputs,) + results[1:]


def maybe_split_sequence_lengths(sequence_length, num_splits, total_length):
  """Validates and splits `sequence_length`, if necessary.

  Returned value must be used in graph for all validations to be executed.

  Args:
    sequence_length: A batch of sequence lengths, either sized `[batch_size]`
      and equal to either 0 or `total_length`, or sized
      `[batch_size, num_splits]`.
    num_splits: The scalar number of splits of the full sequences.
    total_length: The scalar total sequence length (potentially padded).

  Returns:
    sequence_length: If input shape was `[batch_size, num_splits]`, returns the
      same Tensor. Otherwise, returns a Tensor of that shape with each input
      length in the batch divided by `num_splits`.
  Raises:
    ValueError: If `sequence_length` is not shaped `[batch_size]` or
      `[batch_size, num_splits]`.
    tf.errors.InvalidArgumentError: If `sequence_length` is shaped
      `[batch_size]` and all values are not either 0 or `total_length`.
  """
  if sequence_length.shape.ndims == 1:
    if total_length % num_splits != 0:
      raise ValueError(
          '`total_length` must be evenly divisible by `num_splits`.')
    with tf.control_dependencies(
        [tf.Assert(
            tf.reduce_all(
                tf.logical_or(tf.equal(sequence_length, 0),
                              tf.equal(sequence_length, total_length))),
            data=[sequence_length])]):
      sequence_length = (
          tf.tile(tf.expand_dims(sequence_length, axis=1), [1, num_splits]) //
          num_splits)
  elif sequence_length.shape.ndims == 2:
    with tf.control_dependencies([
        tf.assert_less_equal(
            sequence_length,
            tf.constant(total_length // num_splits, tf.int32),
            message='Segment length cannot be more than '
                    '`total_length / num_splits`.')]):
      sequence_length = tf.identity(sequence_length)
    sequence_length.set_shape([sequence_length.shape[0], num_splits])
  else:
    raise ValueError(
        'Sequence lengths must be given as a vector or a 2D Tensor whose '
        'second dimension size matches its initial hierarchical split. Got '
        'shape: %s' % sequence_length.shape.as_list())
  return sequence_length
