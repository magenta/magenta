"""Utilities for "fast" wavenet generation with queues.

For more information, see:

Ramachandran, P., Le Paine, T., Khorrami, P., Babaeizadeh, M.,
Chang, S., Zhang, Y., … Huang, T. (2017).
Fast Generation For Convolutional Autoregressive Models, 1–5.
"""
import tensorflow as tf


def causal_linear(x, n_inputs, n_outputs, name, filter_length, rate, batch_size):
  """Applies dilated convolution using queues.

  Assumes a filter_length of 3.

  Args:
    x: The [mb, time, channels] tensor input.
    n_inputs: The input number of channels.
    n_outputs: The output number of channels.
    name: The variable scope to provide to W and biases.
    filter_length: The length of the convolution, assumed to be 3.
    rate: The rate or dilation
    batch_size: Non-symbolic value for batch_size.

  Returns:
    y: The output of the operation
    (init_1, init_2): Initialization operations for the queues
    (push_1, push_2): Push operations for the queues
  """
  assert(filter_length == 3)

  # create queue
  q_1 = tf.FIFOQueue(
      rate,
      dtypes=tf.float32,
      shapes=(batch_size, n_inputs))
  q_2 = tf.FIFOQueue(
      rate,
      dtypes=tf.float32,
      shapes=(batch_size, n_inputs))
  init_1 = q_1.enqueue_many(
      tf.zeros((rate, batch_size, n_inputs)))
  init_2 = q_2.enqueue_many(
      tf.zeros((rate, batch_size, n_inputs)))
  state_1 = q_1.dequeue()
  push_1 = q_1.enqueue(x)
  state_2 = q_2.dequeue()
  push_2 = q_2.enqueue(state_1)

  # get pretrained weights
  W = tf.get_variable(
      name=name + '/W',
      shape=[1, filter_length, n_inputs, n_outputs],
      dtype=tf.float32)
  b = tf.get_variable(
      name=name + '/biases',
      shape=[n_outputs],
      dtype=tf.float32)
  W_q_2 = tf.slice(W, [0, 0, 0, 0], [-1, 1, -1, -1])
  W_q_1 = tf.slice(W, [0, 1, 0, 0], [-1, 1, -1, -1])
  W_x = tf.slice(W, [0, 2, 0, 0], [-1, 1, -1, -1])

  # perform op w/ cached states
  y = tf.expand_dims(tf.nn.bias_add(
      tf.matmul(state_2, W_q_2[0][0]) +
      tf.matmul(state_1, W_q_1[0][0]) +
      tf.matmul(x, W_x[0][0]),
      b), 0)
  return y, (init_1, init_2), (push_1, push_2)


def linear(x, n_inputs, n_outputs, name):
  """Simple linear layer.

  Args:
    x: The [mb, time, channels] tensor input.
    n_inputs: The input number of channels.
    n_outputs: The output number of channels.
    name: The variable scope to provide to W and biases.

  Returns:
    y: The output of the operation.
  """
  W = tf.get_variable(
      name=name + '/W',
      shape=[1, 1, n_inputs, n_outputs],
      dtype=tf.float32)
  b = tf.get_variable(
      name=name + '/biases',
      shape=[n_outputs],
      dtype=tf.float32)
  return tf.expand_dims(tf.nn.bias_add(tf.matmul(x[0], W[0][0]), b), 0)
