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

"""Tests for forked classes and functions from `tf.contrib.rnn`."""
import itertools

from absl.testing import parameterized
from magenta.contrib import rnn as contrib_rnn
import numpy as np
import tensorflow.compat.v1 as tf

rnn_cell = tf.nn.rnn_cell

tf.disable_eager_execution()

# pylint:disable=invalid-name


class RNNCellTest(tf.test.TestCase):

  def testInputProjectionWrapper(self):
    with self.cached_session() as sess:
      with tf.variable_scope(
          "root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 2])
        m = tf.zeros([1, 3])
        cell = contrib_rnn.InputProjectionWrapper(
            rnn_cell.GRUCell(3), num_proj=3)
        g, new_m = cell(x, m)
        sess.run([tf.global_variables_initializer()])
        res = sess.run([g, new_m], {
            x.name: np.array([[1., 1.]]),
            m.name: np.array([[0.1, 0.1, 0.1]])
        })
        self.assertEqual(res[1].shape, (1, 3))
        # The numbers in results were not calculated, this is just a smoke test.
        self.assertAllClose(res[0], [[0.154605, 0.154605, 0.154605]])

  def testAttentionCellWrapperFailures(self):
    with self.assertRaisesRegexp(
        TypeError, contrib_rnn.ASSERT_LIKE_RNNCELL_ERROR_REGEXP):
      contrib_rnn.AttentionCellWrapper(None, 0)

    num_units = 8
    for state_is_tuple in [False, True]:
      with tf.Graph().as_default():
        lstm_cell = rnn_cell.BasicLSTMCell(
            num_units, state_is_tuple=state_is_tuple)
        with self.assertRaisesRegexp(
            ValueError, "attn_length should be greater than zero, got 0"):
          contrib_rnn.AttentionCellWrapper(
              lstm_cell, 0, state_is_tuple=state_is_tuple)
        with self.assertRaisesRegexp(
            ValueError, "attn_length should be greater than zero, got -1"):
          contrib_rnn.AttentionCellWrapper(
              lstm_cell, -1, state_is_tuple=state_is_tuple)
      with tf.Graph().as_default():
        lstm_cell = rnn_cell.BasicLSTMCell(num_units, state_is_tuple=True)
        with self.assertRaisesRegexp(
            ValueError, "Cell returns tuple of states, but the flag "
            "state_is_tuple is not set. State size is: *"):
          contrib_rnn.AttentionCellWrapper(
              lstm_cell, 4, state_is_tuple=False)

  def testAttentionCellWrapperZeros(self):
    num_units = 8
    attn_length = 16
    batch_size = 3
    input_size = 4
    for state_is_tuple in [False, True]:
      with tf.Graph().as_default():
        with self.cached_session() as sess:
          with tf.variable_scope(
              "state_is_tuple_" + str(state_is_tuple)):
            lstm_cell = rnn_cell.BasicLSTMCell(
                num_units, state_is_tuple=state_is_tuple)
            cell = contrib_rnn.AttentionCellWrapper(
                lstm_cell, attn_length, state_is_tuple=state_is_tuple)
            if state_is_tuple:
              zeros = tf.zeros([batch_size, num_units], dtype=np.float32)
              attn_state_zeros = tf.zeros(
                  [batch_size, attn_length * num_units], dtype=np.float32)
              zero_state = ((zeros, zeros), zeros, attn_state_zeros)
            else:
              zero_state = tf.zeros(
                  [
                      batch_size,
                      num_units * 2 + attn_length * num_units + num_units
                  ],
                  dtype=np.float32)
            inputs = tf.zeros(
                [batch_size, input_size], dtype=tf.float32)
            output, state = cell(inputs, zero_state)
            self.assertEqual(output.get_shape(), [batch_size, num_units])
            if state_is_tuple:
              self.assertEqual(len(state), 3)
              self.assertEqual(len(state[0]), 2)
              self.assertEqual(state[0][0].get_shape(), [batch_size, num_units])
              self.assertEqual(state[0][1].get_shape(), [batch_size, num_units])
              self.assertEqual(state[1].get_shape(), [batch_size, num_units])
              self.assertEqual(state[2].get_shape(),
                               [batch_size, attn_length * num_units])
              tensors = [output] + list(state)
            else:
              self.assertEqual(state.get_shape(), [
                  batch_size,
                  num_units * 2 + num_units + attn_length * num_units
              ])
              tensors = [output, state]
            zero_result = sum(
                [tf.reduce_sum(tf.abs(x)) for x in tensors])
            sess.run(tf.global_variables_initializer())
            self.assertLess(sess.run(zero_result), 1e-6)

  def testAttentionCellWrapperValues(self):
    num_units = 8
    attn_length = 16
    batch_size = 3
    for state_is_tuple in [False, True]:
      with tf.Graph().as_default():
        with self.cached_session() as sess:
          with tf.variable_scope(
              "state_is_tuple_" + str(state_is_tuple)):
            lstm_cell = rnn_cell.BasicLSTMCell(
                num_units, state_is_tuple=state_is_tuple)
            cell = contrib_rnn.AttentionCellWrapper(
                lstm_cell, attn_length, state_is_tuple=state_is_tuple)
            if state_is_tuple:
              zeros = tf.constant(
                  0.1 * np.ones([batch_size, num_units], dtype=np.float32),
                  dtype=tf.float32)
              attn_state_zeros = tf.constant(
                  0.1 * np.ones(
                      [batch_size, attn_length * num_units], dtype=np.float32),
                  dtype=tf.float32)
              zero_state = ((zeros, zeros), zeros, attn_state_zeros)
            else:
              zero_state = tf.constant(
                  0.1 * np.ones(
                      [
                          batch_size,
                          num_units * 2 + num_units + attn_length * num_units
                      ],
                      dtype=np.float32),
                  dtype=tf.float32)
            inputs = tf.constant(
                np.array(
                    [[1., 1., 1., 1.], [2., 2., 2., 2.], [3., 3., 3., 3.]],
                    dtype=np.float32),
                dtype=tf.float32)
            output, state = cell(inputs, zero_state)
            if state_is_tuple:
              concat_state = tf.concat(
                  [state[0][0], state[0][1], state[1], state[2]], 1)
            else:
              concat_state = state
            sess.run(tf.global_variables_initializer())
            output, state = sess.run([output, concat_state])
            # Different inputs so different outputs and states
            for i in range(1, batch_size):
              self.assertGreater(
                  float(np.linalg.norm((output[0, :] - output[i, :]))), 1e-6)
              self.assertGreater(
                  float(np.linalg.norm((state[0, :] - state[i, :]))), 1e-6)

  def _testAttentionCellWrapperCorrectResult(self):
    num_units = 4
    attn_length = 6
    batch_size = 2
    expected_output = np.array(
        [[1.068372, 0.45496, -0.678277, 0.340538],
         [1.018088, 0.378983, -0.572179, 0.268591]],
        dtype=np.float32)
    expected_state = np.array(
        [[
            0.74946702, 0.34681597, 0.26474735, 1.06485605, 0.38465962,
            0.11420801, 0.10272158, 0.30925757, 0.63899988, 0.7181077,
            0.47534478, 0.33715725, 0.58086717, 0.49446869, 0.7641536,
            0.12814975, 0.92231739, 0.89857256, 0.21889746, 0.38442063,
            0.53481543, 0.8876909, 0.45823169, 0.5905602, 0.78038228,
            0.56501579, 0.03971386, 0.09870267, 0.8074435, 0.66821432,
            0.99211812, 0.12295902, 1.14606023, 0.34370938, -0.79251152,
            0.51843399
        ], [
            0.5179342, 0.48682183, -0.25426468, 0.96810579, 0.28809637,
            0.13607743, -0.11446252, 0.26792109, 0.78047138, 0.63460857,
            0.49122369, 0.52007174, 0.73000264, 0.66986895, 0.73576689,
            0.86301267, 0.87887371, 0.35185754, 0.93417215, 0.64732957,
            0.63173044, 0.66627824, 0.53644657, 0.20477486, 0.98458421,
            0.38277245, 0.03746676, 0.92510188, 0.57714164, 0.84932971,
            0.36127412, 0.12125921, 1.1362772, 0.34361625, -0.78150457,
            0.70582712
        ]],
        dtype=np.float32)
    seed = 12345
    tf.set_random_seed(seed)
    rnn_scope = None
    for state_is_tuple in [False, True]:
      with tf.Session() as sess:
        with tf.variable_scope(
            "state_is_tuple",
            reuse=state_is_tuple,
            initializer=tf.glorot_uniform_initializer()):
          lstm_cell = rnn_cell.BasicLSTMCell(
              num_units, state_is_tuple=state_is_tuple)
          cell = contrib_rnn.AttentionCellWrapper(
              lstm_cell, attn_length, state_is_tuple=state_is_tuple)
          # This is legacy behavior to preserve the test.  Weight
          # sharing no longer works by creating a new RNNCell in the
          # same variable scope; so here we restore the scope of the
          # RNNCells after the first use below.
          if rnn_scope is not None:
            (cell._scope, lstm_cell._scope) = rnn_scope  # pylint: disable=protected-access,unpacking-non-sequence
          zeros1 = tf.random_uniform(
              (batch_size, num_units), 0.0, 1.0, seed=seed + 1)
          zeros2 = tf.random_uniform(
              (batch_size, num_units), 0.0, 1.0, seed=seed + 2)
          zeros3 = tf.random_uniform(
              (batch_size, num_units), 0.0, 1.0, seed=seed + 3)
          attn_state_zeros = tf.random_uniform(
              (batch_size, attn_length * num_units), 0.0, 1.0, seed=seed + 4)
          zero_state = ((zeros1, zeros2), zeros3, attn_state_zeros)
          if not state_is_tuple:
            zero_state = tf.concat([
                zero_state[0][0], zero_state[0][1], zero_state[1], zero_state[2]
            ], 1)
          inputs = tf.random_uniform(
              (batch_size, num_units), 0.0, 1.0, seed=seed + 5)
          output, state = cell(inputs, zero_state)
          # This is legacy behavior to preserve the test.  Weight
          # sharing no longer works by creating a new RNNCell in the
          # same variable scope; so here we store the scope of the
          # first RNNCell for reuse above.
          if rnn_scope is None:
            rnn_scope = (cell._scope, lstm_cell._scope)  # pylint: disable=protected-access
          if state_is_tuple:
            state = tf.concat(
                [state[0][0], state[0][1], state[1], state[2]], 1)
          sess.run(tf.global_variables_initializer())
          self.assertAllClose(sess.run(output), expected_output)
          self.assertAllClose(sess.run(state), expected_state)


class StackBidirectionalRNNTest(tf.test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)
    super().setUp()

  def _createStackBidirectionalDynamicRNN(self,
                                          use_gpu,
                                          use_shape,
                                          use_state_tuple,
                                          initial_states_fw=None,
                                          initial_states_bw=None,
                                          scope=None):
    del use_gpu
    del use_state_tuple
    self.layers = [2, 3]
    input_size = 5
    batch_size = 2
    max_length = 8

    initializer = tf.random_uniform_initializer(
        -0.01, 0.01, seed=self._seed)
    sequence_length = tf.placeholder(tf.int64)

    self.cells_fw = [
        rnn_cell.LSTMCell(  # pylint:disable=g-complex-comprehension
            num_units,
            input_size,
            initializer=initializer,
            state_is_tuple=False) for num_units in self.layers
    ]
    self.cells_bw = [
        rnn_cell.LSTMCell(  # pylint:disable=g-complex-comprehension
            num_units,
            input_size,
            initializer=initializer,
            state_is_tuple=False) for num_units in self.layers
    ]

    inputs = max_length * [
        tf.placeholder(
            tf.float32,
            shape=(batch_size, input_size) if use_shape else (None, input_size))
    ]
    inputs_c = tf.stack(inputs)
    inputs_c = tf.transpose(inputs_c, [1, 0, 2])
    outputs, st_fw, st_bw = contrib_rnn.stack_bidirectional_dynamic_rnn(
        self.cells_fw,
        self.cells_bw,
        inputs_c,
        initial_states_fw=initial_states_fw,
        initial_states_bw=initial_states_bw,
        dtype=tf.float32,
        sequence_length=sequence_length,
        scope=scope)

    # Outputs has shape (batch_size, max_length, 2* layer[-1].
    output_shape = [None, max_length, 2 * self.layers[-1]]
    if use_shape:
      output_shape[0] = batch_size

    self.assertAllEqual(outputs.get_shape().as_list(), output_shape)

    input_value = np.random.randn(batch_size, input_size)

    return input_value, inputs, outputs, st_fw, st_bw, sequence_length

  def _testStackBidirectionalDynamicRNN(self, use_gpu, use_shape,
                                        use_state_tuple):
    with self.session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      input_value, inputs, outputs, state_fw, state_bw, sequence_length = (
          self._createStackBidirectionalDynamicRNN(use_gpu, use_shape,
                                                   use_state_tuple))
      tf.global_variables_initializer().run()
      # Run with pre-specified sequence length of 2, 3
      out, s_fw, s_bw = sess.run(
          [outputs, state_fw, state_bw],
          feed_dict={inputs[0]: input_value,
                     sequence_length: [2, 3]})

      # Since the forward and backward LSTM cells were initialized with the
      # same parameters, the forward and backward states of the first layer has
      # to be the same.
      # For the next layers, since the input is a concat of forward and backward
      # outputs of the previous layers the symmetry is broken and the following
      # states and outputs differ.
      # We cannot access the intermediate values between layers but we can
      # check that the forward and backward states of the first layer match.

      self.assertAllClose(s_fw[0], s_bw[0])
      out = np.swapaxes(out, 0, 1)
      # If outputs are not concat between layers the output of the forward
      # and backward would be the same but symmetric.
      # Check that is not the case.
      # Due to depth concatenation (as num_units=3 for both RNNs):
      # - forward output:  out[][][depth] for 0 <= depth < 3
      # - backward output: out[][][depth] for 4 <= depth < 6
      # First sequence in batch is length=2
      # Check that the time=0 forward output is not equal to time=1 backward.
      self.assertNotEqual(out[0][0][0], out[1][0][3])
      self.assertNotEqual(out[0][0][1], out[1][0][4])
      self.assertNotEqual(out[0][0][2], out[1][0][5])
      # Check that the time=1 forward output is not equal to time=0 backward.
      self.assertNotEqual(out[1][0][0], out[0][0][3])
      self.assertNotEqual(out[1][0][1], out[0][0][4])
      self.assertNotEqual(out[1][0][2], out[0][0][5])

      # Second sequence in batch is length=3
      # Check that the time=0 forward output is not equal to time=2 backward.
      self.assertNotEqual(out[0][1][0], out[2][1][3])
      self.assertNotEqual(out[0][1][1], out[2][1][4])
      self.assertNotEqual(out[0][1][2], out[2][1][5])
      # Check that the time=1 forward output is not equal to time=1 backward.
      self.assertNotEqual(out[1][1][0], out[1][1][3])
      self.assertNotEqual(out[1][1][1], out[1][1][4])
      self.assertNotEqual(out[1][1][2], out[1][1][5])
      # Check that the time=2 forward output is not equal to time=0 backward.
      self.assertNotEqual(out[2][1][0], out[0][1][3])
      self.assertNotEqual(out[2][1][1], out[0][1][4])
      self.assertNotEqual(out[2][1][2], out[0][1][5])

  def _testStackBidirectionalDynamicRNNStates(self, use_gpu):

    # Check that the states are correctly initialized.
    # - Create a net and iterate for 3 states. Keep the state (state_3).
    # - Reset states, and iterate for 5 steps. Last state is state_5.
    # - Reset the sets to state_3 and iterate for 2 more steps,
    #   last state will be state_5'.
    # - Check that the state_5 and state_5' (forward and backward) are the
    #   same for the first layer (it does not apply for the second layer since
    #   it has forward-backward dependencies).
    with self.session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      batch_size = 2
      # Create states placeholders.
      initial_states_fw = [
          tf.placeholder(
              tf.float32, shape=(batch_size, layer * 2))
          for layer in self.layers
      ]
      initial_states_bw = [
          tf.placeholder(
              tf.float32, shape=(batch_size, layer * 2))
          for layer in self.layers
      ]
      # Create the net
      input_value, inputs, outputs, state_fw, state_bw, sequence_length = (
          self._createStackBidirectionalDynamicRNN(
              use_gpu,
              use_shape=True,
              use_state_tuple=False,
              initial_states_fw=initial_states_fw,
              initial_states_bw=initial_states_bw))
      tf.global_variables_initializer().run()

      # Run 3 steps.
      feed_dict = {inputs[0]: input_value, sequence_length: [3, 2]}
      # Initialize to empty state.
      for i, layer in enumerate(self.layers):
        feed_dict[initial_states_fw[i]] = np.zeros(
            (batch_size, layer * 2), dtype=np.float32)
        feed_dict[initial_states_bw[i]] = np.zeros(
            (batch_size, layer * 2), dtype=np.float32)
      _, st_3_fw, st_3_bw = sess.run([outputs, state_fw, state_bw],
                                     feed_dict=feed_dict)

      # Reset the net and run 5 steps.
      feed_dict = {inputs[0]: input_value, sequence_length: [5, 3]}
      for i, layer in enumerate(self.layers):
        feed_dict[initial_states_fw[i]] = np.zeros(
            (batch_size, layer * 2), dtype=np.float32)
        feed_dict[initial_states_bw[i]] = np.zeros(
            (batch_size, layer * 2), dtype=np.float32)
      _, st_5_fw, st_5_bw = sess.run([outputs, state_fw, state_bw],
                                     feed_dict=feed_dict)

      # Reset the net to state_3 and run 2 more steps.
      feed_dict = {inputs[0]: input_value, sequence_length: [2, 1]}
      for i, _ in enumerate(self.layers):
        feed_dict[initial_states_fw[i]] = st_3_fw[i]
        feed_dict[initial_states_bw[i]] = st_3_bw[i]
      _, st_5p_fw, st_5p_bw = sess.run([outputs, state_fw, state_bw],
                                       feed_dict=feed_dict)

      # Check that the 3+2 and 5 first layer states.
      self.assertAllEqual(st_5_fw[0], st_5p_fw[0])
      self.assertAllEqual(st_5_bw[0], st_5p_bw[0])

  def testBidirectionalRNN(self):
    # Generate 2^3 option values
    # from [True, True, True] to [False, False, False]
    options = itertools.product([True, False], repeat=3)
    for option in options:
      self._testStackBidirectionalDynamicRNN(
          use_gpu=option[0], use_shape=option[1], use_state_tuple=option[2])
    # Check States.
    self._testStackBidirectionalDynamicRNNStates(use_gpu=False)
    self._testStackBidirectionalDynamicRNNStates(use_gpu=True)

  def _testScope(self, factory, prefix="prefix", use_outer_scope=True):
    # REMARKS: factory(scope) is a function accepting a scope
    #          as an argument, such scope can be None, a string
    #          or a VariableScope instance.
    with self.session(use_gpu=True, graph=tf.Graph()):
      if use_outer_scope:
        with tf.variable_scope(prefix) as scope:
          factory(scope)
      else:
        factory(prefix)

      # check that all the variables names starts with the proper scope.
      tf.global_variables_initializer()
      all_vars = tf.global_variables()
      prefix = prefix or "stack_bidirectional_rnn"
      scope_vars = [v for v in all_vars if v.name.startswith(prefix + "/")]
      tf.logging.info("StackRNN with scope: %s (%s)" %
                      (prefix, "scope" if use_outer_scope else "str"))
      for v in scope_vars:
        tf.logging.info(v.name)
      self.assertEqual(len(scope_vars), len(all_vars))

  def testBidirectionalDynamicRNNScope(self):

    def factory(scope):
      return self._createStackBidirectionalDynamicRNN(
          use_gpu=True, use_shape=True, use_state_tuple=True, scope=scope)

    self._testScope(factory, use_outer_scope=True)
    self._testScope(factory, use_outer_scope=False)
    self._testScope(factory, prefix=None, use_outer_scope=False)


class LSTMBlockCellTest(tf.test.TestCase, parameterized.TestCase):

  TEST_CASES = ({
      "testcase_name": "Fp32",
      "dtype": tf.float32,
      "rtol": 1e-6,
      "atol": 1e-6
  }, {
      "testcase_name": "Fp16",
      "dtype": tf.float16,
      "rtol": 8e-3,
      "atol": 8e-4
  })

  def testNoneDimsWithDynamicRNN(self):
    with self.session(use_gpu=True, graph=tf.Graph()) as sess:
      batch_size = 4
      num_steps = 5
      input_dim = 6
      cell_size = 7

      cell = contrib_rnn.LSTMBlockCell(cell_size)
      x = tf.placeholder(tf.float32, shape=(None, None, input_dim))

      output, _ = tf.nn.dynamic_rnn(
          cell, x, time_major=True, dtype=tf.float32)
      sess.run(tf.global_variables_initializer())
      feed = {}
      feed[x] = np.random.randn(num_steps, batch_size, input_dim)
      sess.run(output, feed)

  def testLSTMBlockCell(self):
    with self.session(use_gpu=True, graph=tf.Graph()) as sess:
      with tf.variable_scope(
          "root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 2])
        m0 = tf.zeros([1, 2])
        m1 = tf.zeros([1, 2])
        m2 = tf.zeros([1, 2])
        m3 = tf.zeros([1, 2])
        g, ((out_m0, out_m1), (out_m2, out_m3)) = rnn_cell.MultiRNNCell(
            [contrib_rnn.LSTMBlockCell(2)
             for _ in range(2)], state_is_tuple=True)(x, ((m0, m1), (m2, m3)))
        sess.run([tf.global_variables_initializer()])
        res = sess.run([g, out_m0, out_m1, out_m2, out_m3], {
            x.name: np.array([[1., 1.]]),
            m0.name: 0.1 * np.ones([1, 2]),
            m1.name: 0.1 * np.ones([1, 2]),
            m2.name: 0.1 * np.ones([1, 2]),
            m3.name: 0.1 * np.ones([1, 2])
        })
        self.assertLen(res, 5)
        self.assertAllClose(res[0], [[0.24024698, 0.24024698]])
        # These numbers are from testBasicLSTMCell and only test c/h.
        self.assertAllClose(res[1], [[0.68967271, 0.68967271]])
        self.assertAllClose(res[2], [[0.44848421, 0.44848421]])
        self.assertAllClose(res[3], [[0.39897051, 0.39897051]])
        self.assertAllClose(res[4], [[0.24024698, 0.24024698]])

  def testCompatibleNames(self):
    with self.session(use_gpu=True, graph=tf.Graph()):
      cell = rnn_cell.LSTMCell(10)
      pcell = rnn_cell.LSTMCell(10, use_peepholes=True)
      inputs = [tf.zeros([4, 5])] * 6
      tf.nn.static_rnn(cell, inputs, dtype=tf.float32, scope="basic")
      tf.nn.static_rnn(pcell, inputs, dtype=tf.float32, scope="peephole")
      basic_names = {
          v.name: v.get_shape()
          for v in tf.trainable_variables()
      }

    with self.session(use_gpu=True, graph=tf.Graph()):
      cell = contrib_rnn.LSTMBlockCell(10)
      pcell = contrib_rnn.LSTMBlockCell(10, use_peephole=True)
      inputs = [tf.zeros([4, 5])] * 6
      tf.nn.static_rnn(cell, inputs, dtype=tf.float32, scope="basic")
      tf.nn.static_rnn(pcell, inputs, dtype=tf.float32, scope="peephole")
      block_names = {
          v.name: v.get_shape()
          for v in tf.trainable_variables()
      }

    self.assertEqual(basic_names, block_names)

  def testLSTMBasicToBlockCell(self):
    with self.session(use_gpu=True) as sess:
      x = tf.zeros([1, 2])
      x_values = np.random.randn(1, 2)

      m0_val = 0.1 * np.ones([1, 2])
      m1_val = -0.1 * np.ones([1, 2])
      m2_val = -0.2 * np.ones([1, 2])
      m3_val = 0.2 * np.ones([1, 2])

      initializer = tf.random_uniform_initializer(
          -0.01, 0.01, seed=19890212)
      with tf.variable_scope("basic", initializer=initializer):
        m0 = tf.zeros([1, 2])
        m1 = tf.zeros([1, 2])
        m2 = tf.zeros([1, 2])
        m3 = tf.zeros([1, 2])
        g, ((out_m0, out_m1), (out_m2, out_m3)) = rnn_cell.MultiRNNCell(
            [rnn_cell.BasicLSTMCell(2, state_is_tuple=True) for _ in range(2)],
            state_is_tuple=True)(x, ((m0, m1), (m2, m3)))
        sess.run([tf.global_variables_initializer()])
        basic_res = sess.run([g, out_m0, out_m1, out_m2, out_m3], {
            x.name: x_values,
            m0.name: m0_val,
            m1.name: m1_val,
            m2.name: m2_val,
            m3.name: m3_val
        })

      with tf.variable_scope("block", initializer=initializer):
        m0 = tf.zeros([1, 2])
        m1 = tf.zeros([1, 2])
        m2 = tf.zeros([1, 2])
        m3 = tf.zeros([1, 2])
        g, ((out_m0, out_m1), (out_m2, out_m3)) = rnn_cell.MultiRNNCell(
            [contrib_rnn.LSTMBlockCell(2)
             for _ in range(2)], state_is_tuple=True)(x, ((m0, m1), (m2, m3)))
        sess.run([tf.global_variables_initializer()])
        block_res = sess.run([g, out_m0, out_m1, out_m2, out_m3], {
            x.name: x_values,
            m0.name: m0_val,
            m1.name: m1_val,
            m2.name: m2_val,
            m3.name: m3_val
        })

      self.assertEqual(len(basic_res), len(block_res))
      for basic, block in zip(basic_res, block_res):
        self.assertAllClose(basic, block)

  def testLSTMBasicToBlockCellPeeping(self):
    with self.session(use_gpu=True) as sess:
      x = tf.zeros([1, 2])
      x_values = np.random.randn(1, 2)

      m0_val = 0.1 * np.ones([1, 2])
      m1_val = -0.1 * np.ones([1, 2])
      m2_val = -0.2 * np.ones([1, 2])
      m3_val = 0.2 * np.ones([1, 2])

      initializer = tf.random_uniform_initializer(
          -0.01, 0.01, seed=19890212)
      with tf.variable_scope("basic", initializer=initializer):
        m0 = tf.zeros([1, 2])
        m1 = tf.zeros([1, 2])
        m2 = tf.zeros([1, 2])
        m3 = tf.zeros([1, 2])
        g, ((out_m0, out_m1), (out_m2, out_m3)) = rnn_cell.MultiRNNCell(
            [
                rnn_cell.LSTMCell(2, use_peepholes=True, state_is_tuple=True)
                for _ in range(2)
            ],
            state_is_tuple=True)(x, ((m0, m1), (m2, m3)))
        sess.run([tf.global_variables_initializer()])
        basic_res = sess.run([g, out_m0, out_m1, out_m2, out_m3], {
            x.name: x_values,
            m0.name: m0_val,
            m1.name: m1_val,
            m2.name: m2_val,
            m3.name: m3_val
        })

      with tf.variable_scope("block", initializer=initializer):
        m0 = tf.zeros([1, 2])
        m1 = tf.zeros([1, 2])
        m2 = tf.zeros([1, 2])
        m3 = tf.zeros([1, 2])
        g, ((out_m0, out_m1), (out_m2, out_m3)) = rnn_cell.MultiRNNCell(
            [contrib_rnn.LSTMBlockCell(2, use_peephole=True) for _ in range(2)],
            state_is_tuple=True)(x, ((m0, m1), (m2, m3)))
        sess.run([tf.global_variables_initializer()])
        block_res = sess.run([g, out_m0, out_m1, out_m2, out_m3], {
            x.name: x_values,
            m0.name: m0_val,
            m1.name: m1_val,
            m2.name: m2_val,
            m3.name: m3_val
        })

      self.assertEqual(len(basic_res), len(block_res))
      for basic, block in zip(basic_res, block_res):
        self.assertAllClose(basic, block)


class LayerNormBasicLSTMCellTest(tf.test.TestCase):

  # NOTE: all the values in the current test case have been calculated.

  def testBasicLSTMCell(self):
    with self.cached_session() as sess:
      with tf.variable_scope(
          "root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 2])
        c0 = tf.zeros([1, 2])
        h0 = tf.zeros([1, 2])
        state0 = rnn_cell.LSTMStateTuple(c0, h0)
        c1 = tf.zeros([1, 2])
        h1 = tf.zeros([1, 2])
        state1 = rnn_cell.LSTMStateTuple(c1, h1)
        state = (state0, state1)
        single_cell = lambda: contrib_rnn.LayerNormBasicLSTMCell(2)
        cell = rnn_cell.MultiRNNCell([single_cell() for _ in range(2)])
        g, out_m = cell(x, state)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(
            [g, out_m], {
                x.name: np.array([[1., 1.]]),
                c0.name: 0.1 * np.asarray([[0, 1]]),
                h0.name: 0.1 * np.asarray([[2, 3]]),
                c1.name: 0.1 * np.asarray([[4, 5]]),
                h1.name: 0.1 * np.asarray([[6, 7]]),
            })

        expected_h = np.array([[-0.38079708, 0.38079708]])
        expected_state0_c = np.array([[-1.0, 1.0]])
        expected_state0_h = np.array([[-0.38079708, 0.38079708]])
        expected_state1_c = np.array([[-1.0, 1.0]])
        expected_state1_h = np.array([[-0.38079708, 0.38079708]])

        actual_h = res[0]
        actual_state0_c = res[1][0].c
        actual_state0_h = res[1][0].h
        actual_state1_c = res[1][1].c
        actual_state1_h = res[1][1].h

        self.assertAllClose(actual_h, expected_h, 1e-5)
        self.assertAllClose(expected_state0_c, actual_state0_c, 1e-5)
        self.assertAllClose(expected_state0_h, actual_state0_h, 1e-5)
        self.assertAllClose(expected_state1_c, actual_state1_c, 1e-5)
        self.assertAllClose(expected_state1_h, actual_state1_h, 1e-5)

      with tf.variable_scope(
          "other", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros(
            [1, 3])  # Test BasicLSTMCell with input_size != num_units.
        c = tf.zeros([1, 2])
        h = tf.zeros([1, 2])
        state = rnn_cell.LSTMStateTuple(c, h)
        cell = contrib_rnn.LayerNormBasicLSTMCell(2)
        g, out_m = cell(x, state)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(
            [g, out_m], {
                x.name: np.array([[1., 1., 1.]]),
                c.name: 0.1 * np.asarray([[0, 1]]),
                h.name: 0.1 * np.asarray([[2, 3]]),
            })

        expected_h = np.array([[-0.38079708, 0.38079708]])
        expected_c = np.array([[-1.0, 1.0]])
        self.assertEqual(len(res), 2)
        self.assertAllClose(res[0], expected_h, 1e-5)
        self.assertAllClose(res[1].c, expected_c, 1e-5)
        self.assertAllClose(res[1].h, expected_h, 1e-5)

  def testBasicLSTMCellWithoutNorm(self):
    """Tests that BasicLSTMCell with layer_norm=False."""
    with self.cached_session() as sess:
      with tf.variable_scope(
          "root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 2])
        c0 = tf.zeros([1, 2])
        h0 = tf.zeros([1, 2])
        state0 = rnn_cell.LSTMStateTuple(c0, h0)
        c1 = tf.zeros([1, 2])
        h1 = tf.zeros([1, 2])
        state1 = rnn_cell.LSTMStateTuple(c1, h1)
        state = (state0, state1)
        single_cell = lambda: contrib_rnn.LayerNormBasicLSTMCell(2, layer_norm=False)  # pylint: disable=line-too-long
        cell = rnn_cell.MultiRNNCell([single_cell() for _ in range(2)])
        g, out_m = cell(x, state)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(
            [g, out_m], {
                x.name: np.array([[1., 1.]]),
                c0.name: 0.1 * np.asarray([[0, 1]]),
                h0.name: 0.1 * np.asarray([[2, 3]]),
                c1.name: 0.1 * np.asarray([[4, 5]]),
                h1.name: 0.1 * np.asarray([[6, 7]]),
            })

        expected_h = np.array([[0.70230919, 0.72581059]])
        expected_state0_c = np.array([[0.8020075, 0.89599884]])
        expected_state0_h = np.array([[0.56668288, 0.60858738]])
        expected_state1_c = np.array([[1.17500675, 1.26892781]])
        expected_state1_h = np.array([[0.70230919, 0.72581059]])

        actual_h = res[0]
        actual_state0_c = res[1][0].c
        actual_state0_h = res[1][0].h
        actual_state1_c = res[1][1].c
        actual_state1_h = res[1][1].h

        self.assertAllClose(actual_h, expected_h, 1e-5)
        self.assertAllClose(expected_state0_c, actual_state0_c, 1e-5)
        self.assertAllClose(expected_state0_h, actual_state0_h, 1e-5)
        self.assertAllClose(expected_state1_c, actual_state1_c, 1e-5)
        self.assertAllClose(expected_state1_h, actual_state1_h, 1e-5)

      with tf.variable_scope(
          "other", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros(
            [1, 3])  # Test BasicLSTMCell with input_size != num_units.
        c = tf.zeros([1, 2])
        h = tf.zeros([1, 2])
        state = rnn_cell.LSTMStateTuple(c, h)
        cell = contrib_rnn.LayerNormBasicLSTMCell(2, layer_norm=False)
        g, out_m = cell(x, state)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(
            [g, out_m], {
                x.name: np.array([[1., 1., 1.]]),
                c.name: 0.1 * np.asarray([[0, 1]]),
                h.name: 0.1 * np.asarray([[2, 3]]),
            })

        expected_h = np.array([[0.64121795, 0.68166804]])
        expected_c = np.array([[0.88477188, 0.98103917]])
        self.assertEqual(len(res), 2)
        self.assertAllClose(res[0], expected_h, 1e-5)
        self.assertAllClose(res[1].c, expected_c, 1e-5)
        self.assertAllClose(res[1].h, expected_h, 1e-5)

  def testBasicLSTMCellWithStateTuple(self):
    with self.cached_session() as sess:
      with tf.variable_scope(
          "root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 2])
        c0 = tf.zeros([1, 2])
        h0 = tf.zeros([1, 2])
        state0 = rnn_cell.LSTMStateTuple(c0, h0)
        c1 = tf.zeros([1, 2])
        h1 = tf.zeros([1, 2])
        state1 = rnn_cell.LSTMStateTuple(c1, h1)
        cell = rnn_cell.MultiRNNCell(
            [contrib_rnn.LayerNormBasicLSTMCell(2) for _ in range(2)])
        h, (s0, s1) = cell(x, (state0, state1))
        sess.run([tf.global_variables_initializer()])
        res = sess.run(
            [h, s0, s1], {
                x.name: np.array([[1., 1.]]),
                c0.name: 0.1 * np.asarray([[0, 1]]),
                h0.name: 0.1 * np.asarray([[2, 3]]),
                c1.name: 0.1 * np.asarray([[4, 5]]),
                h1.name: 0.1 * np.asarray([[6, 7]]),
            })

        expected_h = np.array([[-0.38079708, 0.38079708]])
        expected_h0 = np.array([[-0.38079708, 0.38079708]])
        expected_c0 = np.array([[-1.0, 1.0]])
        expected_h1 = np.array([[-0.38079708, 0.38079708]])
        expected_c1 = np.array([[-1.0, 1.0]])

        self.assertEqual(len(res), 3)
        self.assertAllClose(res[0], expected_h, 1e-5)
        self.assertAllClose(res[1].c, expected_c0, 1e-5)
        self.assertAllClose(res[1].h, expected_h0, 1e-5)
        self.assertAllClose(res[2].c, expected_c1, 1e-5)
        self.assertAllClose(res[2].h, expected_h1, 1e-5)

  def testBasicLSTMCellWithStateTupleLayerNorm(self):
    """The results of LSTMCell and LayerNormBasicLSTMCell should be the same."""
    with self.cached_session() as sess:
      with tf.variable_scope(
          "root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 2])
        c0 = tf.zeros([1, 2])
        h0 = tf.zeros([1, 2])
        state0 = rnn_cell.LSTMStateTuple(c0, h0)
        c1 = tf.zeros([1, 2])
        h1 = tf.zeros([1, 2])
        state1 = rnn_cell.LSTMStateTuple(c1, h1)
        cell = rnn_cell.MultiRNNCell([
            contrib_rnn.LayerNormBasicLSTMCell(
                2, layer_norm=True, norm_gain=1.0, norm_shift=0.0)
            for _ in range(2)
        ])
        h, (s0, s1) = cell(x, (state0, state1))
        sess.run([tf.global_variables_initializer()])
        res = sess.run(
            [h, s0, s1], {
                x.name: np.array([[1., 1.]]),
                c0.name: 0.1 * np.asarray([[0, 1]]),
                h0.name: 0.1 * np.asarray([[2, 3]]),
                c1.name: 0.1 * np.asarray([[4, 5]]),
                h1.name: 0.1 * np.asarray([[6, 7]]),
            })

        expected_h = np.array([[-0.38079708, 0.38079708]])
        expected_h0 = np.array([[-0.38079708, 0.38079708]])
        expected_c0 = np.array([[-1.0, 1.0]])
        expected_h1 = np.array([[-0.38079708, 0.38079708]])
        expected_c1 = np.array([[-1.0, 1.0]])

        self.assertEqual(len(res), 3)
        self.assertAllClose(res[0], expected_h, 1e-5)
        self.assertAllClose(res[1].c, expected_c0, 1e-5)
        self.assertAllClose(res[1].h, expected_h0, 1e-5)
        self.assertAllClose(res[2].c, expected_c1, 1e-5)
        self.assertAllClose(res[2].h, expected_h1, 1e-5)

  def testBasicLSTMCellWithDropout(self):

    def _is_close(x, y, digits=4):
      delta = x - y
      return delta < 10**(-digits)

    def _is_close_in(x, items, digits=4):
      for i in items:
        if _is_close(x, i, digits):
          return True
      return False

    keep_prob = 0.5
    c_high = 2.9998924946
    c_low = 0.999983298578
    h_low = 0.761552567265
    h_high = 0.995008519604
    num_units = 5
    allowed_low = [1, 2, 3]

    with self.cached_session() as sess:
      with tf.variable_scope(
          "other", initializer=tf.constant_initializer(1)):
        x = tf.zeros([1, 5])
        c = tf.zeros([1, 5])
        h = tf.zeros([1, 5])
        state = rnn_cell.LSTMStateTuple(c, h)
        cell = contrib_rnn.LayerNormBasicLSTMCell(
            num_units, layer_norm=False, dropout_keep_prob=keep_prob)

        g, s = cell(x, state)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(
            [g, s], {
                x.name: np.ones([1, 5]),
                c.name: np.ones([1, 5]),
                h.name: np.ones([1, 5]),
            })

        # Since the returned tensors are of size [1,n]
        # get the first component right now.
        actual_h = res[0][0]
        actual_state_c = res[1].c[0]
        actual_state_h = res[1].h[0]

        # For each item in `c` (the cell inner state) check that
        # it is equal to one of the allowed values `c_high` (not
        # dropped out) or `c_low` (dropped out) and verify that the
        # corresponding item in `h` (the cell activation) is coherent.
        # Count the dropped activations and check that their number is
        # coherent with the dropout probability.
        dropped_count = 0
        self.assertTrue((actual_h == actual_state_h).all())
        for citem, hitem in zip(actual_state_c, actual_state_h):
          self.assertTrue(_is_close_in(citem, [c_low, c_high]))
          if _is_close(citem, c_low):
            self.assertTrue(_is_close(hitem, h_low))
            dropped_count += 1
          elif _is_close(citem, c_high):
            self.assertTrue(_is_close(hitem, h_high))
        self.assertIn(dropped_count, allowed_low)

if __name__ == "__main__":
  tf.test.main()
