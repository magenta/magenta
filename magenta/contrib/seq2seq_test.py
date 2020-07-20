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
"""Tests for magenta.contrib.seq2seq."""

from magenta.contrib import seq2seq
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class DynamicDecodeRNNTest(tf.test.TestCase):

  def _testDynamicDecodeRNN(self, time_major, maximum_iterations=None):  # pylint:disable=invalid-name
    sequence_length = [3, 4, 3, 1, 0]
    batch_size = 5
    max_time = 8
    input_depth = 7
    cell_depth = 10
    max_out = max(sequence_length)

    with self.session(use_gpu=True) as sess:
      if time_major:
        inputs = np.random.randn(max_time, batch_size,
                                 input_depth).astype(np.float32)
      else:
        inputs = np.random.randn(batch_size, max_time,
                                 input_depth).astype(np.float32)
      cell = tf.nn.rnn_cell.LSTMCell(cell_depth)
      helper = seq2seq.TrainingHelper(
          inputs, sequence_length, time_major=time_major)
      my_decoder = seq2seq.BasicDecoder(
          cell=cell,
          helper=helper,
          initial_state=cell.zero_state(
              dtype=tf.float32, batch_size=batch_size))

      final_outputs, final_state, final_sequence_length = (
          seq2seq.dynamic_decode(my_decoder, output_time_major=time_major,
                                 maximum_iterations=maximum_iterations))

      def _t(shape):
        if time_major:
          return (shape[1], shape[0]) + shape[2:]
        return shape

      self.assertIsInstance(final_outputs, seq2seq.BasicDecoderOutput)
      self.assertIsInstance(final_state, tf.nn.rnn_cell.LSTMStateTuple)

      self.assertEqual(
          (batch_size,),
          tuple(final_sequence_length.get_shape().as_list()))
      self.assertEqual(
          _t((batch_size, None, cell_depth)),
          tuple(final_outputs.rnn_output.get_shape().as_list()))
      self.assertEqual(
          _t((batch_size, None)),
          tuple(final_outputs.sample_id.get_shape().as_list()))

      sess.run(tf.global_variables_initializer())
      sess_results = sess.run({
          "final_outputs": final_outputs,
          "final_state": final_state,
          "final_sequence_length": final_sequence_length,
      })

      # Mostly a smoke test
      time_steps = max_out
      expected_length = sequence_length
      if maximum_iterations is not None:
        time_steps = min(max_out, maximum_iterations)
        expected_length = [min(x, maximum_iterations) for x in expected_length]
      self.assertEqual(
          _t((batch_size, time_steps, cell_depth)),
          sess_results["final_outputs"].rnn_output.shape)
      self.assertEqual(
          _t((batch_size, time_steps)),
          sess_results["final_outputs"].sample_id.shape)
      self.assertCountEqual(expected_length,
                            sess_results["final_sequence_length"])

  def testDynamicDecodeRNNBatchMajor(self):
    self._testDynamicDecodeRNN(time_major=False)

  def testDynamicDecodeRNNTimeMajor(self):
    self._testDynamicDecodeRNN(time_major=True)

  def testDynamicDecodeRNNZeroMaxIters(self):
    self._testDynamicDecodeRNN(time_major=True, maximum_iterations=0)

  def testDynamicDecodeRNNOneMaxIter(self):
    self._testDynamicDecodeRNN(time_major=True, maximum_iterations=1)

  def _testDynamicDecodeRNNWithTrainingHelperMatchesDynamicRNN(  # pylint:disable=invalid-name
      self, use_sequence_length):
    sequence_length = [3, 4, 3, 1, 0]
    batch_size = 5
    max_time = 8
    input_depth = 7
    cell_depth = 10
    max_out = max(sequence_length)

    with self.session(use_gpu=True) as sess:
      inputs = np.random.randn(batch_size, max_time,
                               input_depth).astype(np.float32)

      cell = tf.nn.rnn_cell.LSTMCell(cell_depth)
      zero_state = cell.zero_state(dtype=tf.float32, batch_size=batch_size)
      helper = seq2seq.TrainingHelper(inputs, sequence_length)
      my_decoder = seq2seq.BasicDecoder(
          cell=cell, helper=helper, initial_state=zero_state)

      # Match the variable scope of dynamic_rnn below so we end up
      # using the same variables
      with tf.variable_scope("root") as scope:
        final_decoder_outputs, final_decoder_state, _ = seq2seq.dynamic_decode(
            my_decoder,
            # impute_finished=True ensures outputs and final state
            # match those of dynamic_rnn called with sequence_length not None
            impute_finished=use_sequence_length,
            scope=scope)

      with tf.variable_scope(scope, reuse=True) as scope:
        final_rnn_outputs, final_rnn_state = tf.nn.dynamic_rnn(
            cell,
            inputs,
            sequence_length=sequence_length if use_sequence_length else None,
            initial_state=zero_state,
            scope=scope)

      sess.run(tf.global_variables_initializer())
      sess_results = sess.run({
          "final_decoder_outputs": final_decoder_outputs,
          "final_decoder_state": final_decoder_state,
          "final_rnn_outputs": final_rnn_outputs,
          "final_rnn_state": final_rnn_state
      })

      # Decoder only runs out to max_out; ensure values are identical
      # to dynamic_rnn, which also zeros out outputs and passes along state.
      self.assertAllClose(sess_results["final_decoder_outputs"].rnn_output,
                          sess_results["final_rnn_outputs"][:, 0:max_out, :])
      if use_sequence_length:
        self.assertAllClose(sess_results["final_decoder_state"],
                            sess_results["final_rnn_state"])

  def testDynamicDecodeRNNWithTrainingHelperMatchesDynamicRNNWithSeqLen(self):
    self._testDynamicDecodeRNNWithTrainingHelperMatchesDynamicRNN(
        use_sequence_length=True)

  def testDynamicDecodeRNNWithTrainingHelperMatchesDynamicRNNNoSeqLen(self):
    self._testDynamicDecodeRNNWithTrainingHelperMatchesDynamicRNN(
        use_sequence_length=False)


class BasicDecoderTest(tf.test.TestCase):

  def _testStepWithTrainingHelper(self, use_output_layer):  # pylint:disable=invalid-name
    sequence_length = [3, 4, 3, 1, 0]
    batch_size = 5
    max_time = 8
    input_depth = 7
    cell_depth = 10
    output_layer_depth = 3

    with self.session(use_gpu=True) as sess:
      inputs = np.random.randn(batch_size, max_time,
                               input_depth).astype(np.float32)
      cell = tf.nn.rnn_cell.LSTMCell(cell_depth)
      helper = seq2seq.TrainingHelper(
          inputs, sequence_length, time_major=False)
      if use_output_layer:
        output_layer = tf.layers.Dense(output_layer_depth, use_bias=False)
        expected_output_depth = output_layer_depth
      else:
        output_layer = None
        expected_output_depth = cell_depth
      my_decoder = seq2seq.BasicDecoder(
          cell=cell,
          helper=helper,
          initial_state=cell.zero_state(
              dtype=tf.float32, batch_size=batch_size),
          output_layer=output_layer)
      output_size = my_decoder.output_size
      output_dtype = my_decoder.output_dtype
      self.assertEqual(
          seq2seq.BasicDecoderOutput(expected_output_depth,
                                     tf.TensorShape([])),
          output_size)
      self.assertEqual(
          seq2seq.BasicDecoderOutput(tf.float32, tf.int32),
          output_dtype)

      (first_finished, first_inputs, first_state) = my_decoder.initialize()
      (step_outputs, step_state, step_next_inputs,
       step_finished) = my_decoder.step(
           tf.constant(0), first_inputs, first_state)
      batch_size_t = my_decoder.batch_size

      self.assertIsInstance(first_state, tf.nn.rnn_cell.LSTMStateTuple)
      self.assertIsInstance(step_state, tf.nn.rnn_cell.LSTMStateTuple)
      self.assertIsInstance(step_outputs, seq2seq.BasicDecoderOutput)
      self.assertEqual((batch_size, expected_output_depth),
                       step_outputs[0].get_shape())
      self.assertEqual((batch_size,), step_outputs[1].get_shape())
      self.assertEqual((batch_size, cell_depth), first_state[0].get_shape())
      self.assertEqual((batch_size, cell_depth), first_state[1].get_shape())
      self.assertEqual((batch_size, cell_depth), step_state[0].get_shape())
      self.assertEqual((batch_size, cell_depth), step_state[1].get_shape())

      if use_output_layer:
        # The output layer was accessed
        self.assertEqual(len(output_layer.variables), 1)

      sess.run(tf.global_variables_initializer())
      sess_results = sess.run({
          "batch_size": batch_size_t,
          "first_finished": first_finished,
          "first_inputs": first_inputs,
          "first_state": first_state,
          "step_outputs": step_outputs,
          "step_state": step_state,
          "step_next_inputs": step_next_inputs,
          "step_finished": step_finished
      })

      self.assertAllEqual([False, False, False, False, True],
                          sess_results["first_finished"])
      self.assertAllEqual([False, False, False, True, True],
                          sess_results["step_finished"])
      self.assertEqual(output_dtype.sample_id,
                       sess_results["step_outputs"].sample_id.dtype)
      self.assertAllEqual(
          np.argmax(sess_results["step_outputs"].rnn_output, -1),
          sess_results["step_outputs"].sample_id)

  def testStepWithTrainingHelperNoOutputLayer(self):
    self._testStepWithTrainingHelper(use_output_layer=False)

  def testStepWithTrainingHelperWithOutputLayer(self):
    self._testStepWithTrainingHelper(use_output_layer=True)

  def _testStepWithScheduledOutputTrainingHelper(  # pylint:disable=invalid-name
      self, sampling_probability, use_next_inputs_fn, use_auxiliary_inputs):
    sequence_length = [3, 4, 3, 1, 0]
    batch_size = 5
    max_time = 8
    input_depth = 7
    cell_depth = input_depth
    if use_auxiliary_inputs:
      auxiliary_input_depth = 4
      auxiliary_inputs = np.random.randn(
          batch_size, max_time, auxiliary_input_depth).astype(np.float32)
    else:
      auxiliary_inputs = None

    with self.session(use_gpu=True) as sess:
      inputs = np.random.randn(batch_size, max_time,
                               input_depth).astype(np.float32)
      cell = tf.nn.rnn_cell.LSTMCell(cell_depth)
      sampling_probability = tf.constant(sampling_probability)

      if use_next_inputs_fn:
        def next_inputs_fn(outputs):
          # Use deterministic function for test.
          samples = tf.argmax(outputs, axis=1)
          return tf.one_hot(samples, cell_depth, dtype=tf.float32)
      else:
        next_inputs_fn = None

      helper = seq2seq.ScheduledOutputTrainingHelper(
          inputs=inputs,
          sequence_length=sequence_length,
          sampling_probability=sampling_probability,
          time_major=False,
          next_inputs_fn=next_inputs_fn,
          auxiliary_inputs=auxiliary_inputs)

      my_decoder = seq2seq.BasicDecoder(
          cell=cell,
          helper=helper,
          initial_state=cell.zero_state(
              dtype=tf.float32, batch_size=batch_size))

      output_size = my_decoder.output_size
      output_dtype = my_decoder.output_dtype
      self.assertEqual(
          seq2seq.BasicDecoderOutput(cell_depth, tf.TensorShape([])),
          output_size)
      self.assertEqual(
          seq2seq.BasicDecoderOutput(tf.float32, tf.int32),
          output_dtype)

      (first_finished, first_inputs, first_state) = my_decoder.initialize()
      (step_outputs, step_state, step_next_inputs,
       step_finished) = my_decoder.step(
           tf.constant(0), first_inputs, first_state)

      if use_next_inputs_fn:
        output_after_next_inputs_fn = next_inputs_fn(step_outputs.rnn_output)

      batch_size_t = my_decoder.batch_size

      self.assertIsInstance(first_state, tf.nn.rnn_cell.LSTMStateTuple)
      self.assertIsInstance(step_state, tf.nn.rnn_cell.LSTMStateTuple)
      self.assertIsInstance(step_outputs, seq2seq.BasicDecoderOutput)
      self.assertEqual((batch_size, cell_depth), step_outputs[0].get_shape())
      self.assertEqual((batch_size,), step_outputs[1].get_shape())
      self.assertEqual((batch_size, cell_depth), first_state[0].get_shape())
      self.assertEqual((batch_size, cell_depth), first_state[1].get_shape())
      self.assertEqual((batch_size, cell_depth), step_state[0].get_shape())
      self.assertEqual((batch_size, cell_depth), step_state[1].get_shape())

      sess.run(tf.global_variables_initializer())

      fetches = {
          "batch_size": batch_size_t,
          "first_finished": first_finished,
          "first_inputs": first_inputs,
          "first_state": first_state,
          "step_outputs": step_outputs,
          "step_state": step_state,
          "step_next_inputs": step_next_inputs,
          "step_finished": step_finished
      }
      if use_next_inputs_fn:
        fetches["output_after_next_inputs_fn"] = output_after_next_inputs_fn

      sess_results = sess.run(fetches)

      self.assertAllEqual([False, False, False, False, True],
                          sess_results["first_finished"])
      self.assertAllEqual([False, False, False, True, True],
                          sess_results["step_finished"])

      sample_ids = sess_results["step_outputs"].sample_id
      self.assertEqual(output_dtype.sample_id, sample_ids.dtype)
      batch_where_not_sampling = np.where(np.logical_not(sample_ids))
      batch_where_sampling = np.where(sample_ids)

      auxiliary_inputs_to_concat = (
          auxiliary_inputs[:, 1] if use_auxiliary_inputs else
          np.array([]).reshape(batch_size, 0).astype(np.float32))

      expected_next_sampling_inputs = np.concatenate(
          (sess_results["output_after_next_inputs_fn"][batch_where_sampling]
           if use_next_inputs_fn else
           sess_results["step_outputs"].rnn_output[batch_where_sampling],
           auxiliary_inputs_to_concat[batch_where_sampling]),
          axis=-1)
      self.assertAllClose(
          sess_results["step_next_inputs"][batch_where_sampling],
          expected_next_sampling_inputs)

      self.assertAllClose(
          sess_results["step_next_inputs"][batch_where_not_sampling],
          np.concatenate(
              (np.squeeze(inputs[batch_where_not_sampling, 1], axis=0),
               auxiliary_inputs_to_concat[batch_where_not_sampling]),
              axis=-1))

  def testStepWithScheduledOutputTrainingHelperWithoutNextInputsFnOrAuxInputs(
      self):
    self._testStepWithScheduledOutputTrainingHelper(
        sampling_probability=0.5, use_next_inputs_fn=False,
        use_auxiliary_inputs=False)

  def testStepWithScheduledOutputTrainingHelperWithNextInputsFn(self):
    self._testStepWithScheduledOutputTrainingHelper(
        sampling_probability=0.5, use_next_inputs_fn=True,
        use_auxiliary_inputs=False)

  def testStepWithScheduledOutputTrainingHelperWithAuxiliaryInputs(self):
    self._testStepWithScheduledOutputTrainingHelper(
        sampling_probability=0.5, use_next_inputs_fn=False,
        use_auxiliary_inputs=True)

  def testStepWithScheduledOutputTrainingHelperWithNextInputsFnAndAuxInputs(
      self):
    self._testStepWithScheduledOutputTrainingHelper(
        sampling_probability=0.5, use_next_inputs_fn=True,
        use_auxiliary_inputs=True)

  def testStepWithScheduledOutputTrainingHelperWithNoSampling(self):
    self._testStepWithScheduledOutputTrainingHelper(
        sampling_probability=0.0, use_next_inputs_fn=True,
        use_auxiliary_inputs=True)

  def testStepWithInferenceHelperCategorical(self):
    batch_size = 5
    vocabulary_size = 7
    cell_depth = vocabulary_size
    start_token = 0
    end_token = 6

    start_inputs = tf.one_hot(
        np.ones(batch_size) * start_token,
        vocabulary_size)

    # The sample function samples categorically from the logits.
    sample_fn = lambda x: seq2seq.categorical_sample(logits=x)
    # The next inputs are a one-hot encoding of the sampled labels.
    next_inputs_fn = (
        lambda x: tf.one_hot(x, vocabulary_size, dtype=tf.float32))
    end_fn = lambda sample_ids: tf.equal(sample_ids, end_token)

    with self.session(use_gpu=True) as sess:
      with tf.variable_scope(
          "testStepWithInferenceHelper",
          initializer=tf.constant_initializer(0.01)):
        cell = tf.nn.rnn_cell.LSTMCell(vocabulary_size)
        helper = seq2seq.InferenceHelper(
            sample_fn, sample_shape=(), sample_dtype=tf.int32,
            start_inputs=start_inputs, end_fn=end_fn,
            next_inputs_fn=next_inputs_fn)
        my_decoder = seq2seq.BasicDecoder(
            cell=cell,
            helper=helper,
            initial_state=cell.zero_state(
                dtype=tf.float32, batch_size=batch_size))
        output_size = my_decoder.output_size
        output_dtype = my_decoder.output_dtype
        self.assertEqual(
            seq2seq.BasicDecoderOutput(cell_depth, tf.TensorShape([])),
            output_size)
        self.assertEqual(
            seq2seq.BasicDecoderOutput(tf.float32, tf.int32),
            output_dtype)

        (first_finished, first_inputs, first_state) = my_decoder.initialize()
        (step_outputs, step_state, step_next_inputs,
         step_finished) = my_decoder.step(
             tf.constant(0), first_inputs, first_state)
        batch_size_t = my_decoder.batch_size

        self.assertIsInstance(first_state, tf.nn.rnn_cell.LSTMStateTuple)
        self.assertIsInstance(step_state, tf.nn.rnn_cell.LSTMStateTuple)
        self.assertIsInstance(step_outputs, seq2seq.BasicDecoderOutput)
        self.assertEqual((batch_size, cell_depth), step_outputs[0].get_shape())
        self.assertEqual((batch_size,), step_outputs[1].get_shape())
        self.assertEqual((batch_size, cell_depth), first_state[0].get_shape())
        self.assertEqual((batch_size, cell_depth), first_state[1].get_shape())
        self.assertEqual((batch_size, cell_depth), step_state[0].get_shape())
        self.assertEqual((batch_size, cell_depth), step_state[1].get_shape())

        sess.run(tf.global_variables_initializer())
        sess_results = sess.run({
            "batch_size": batch_size_t,
            "first_finished": first_finished,
            "first_inputs": first_inputs,
            "first_state": first_state,
            "step_outputs": step_outputs,
            "step_state": step_state,
            "step_next_inputs": step_next_inputs,
            "step_finished": step_finished
        })

        sample_ids = sess_results["step_outputs"].sample_id
        self.assertEqual(output_dtype.sample_id, sample_ids.dtype)
        expected_step_finished = (sample_ids == end_token)
        expected_step_next_inputs = np.zeros((batch_size, vocabulary_size))
        expected_step_next_inputs[np.arange(batch_size), sample_ids] = 1.0
        self.assertAllEqual(expected_step_finished,
                            sess_results["step_finished"])
        self.assertAllEqual(expected_step_next_inputs,
                            sess_results["step_next_inputs"])

  def testStepWithInferenceHelperMultilabel(self):
    batch_size = 5
    vocabulary_size = 7
    cell_depth = vocabulary_size
    start_token = 0
    end_token = 6

    start_inputs = tf.one_hot(
        np.ones(batch_size) * start_token,
        vocabulary_size)

    # The sample function samples independent bernoullis from the logits.
    sample_fn = (
        lambda x: seq2seq.bernoulli_sample(logits=x, dtype=tf.bool))
    # The next inputs are a one-hot encoding of the sampled labels.
    next_inputs_fn = tf.to_float
    end_fn = lambda sample_ids: sample_ids[:, end_token]

    with self.session(use_gpu=True) as sess:
      with tf.variable_scope(
          "testStepWithInferenceHelper",
          initializer=tf.constant_initializer(0.01)):
        cell = tf.nn.rnn_cell.LSTMCell(vocabulary_size)
        helper = seq2seq.InferenceHelper(
            sample_fn, sample_shape=[cell_depth], sample_dtype=tf.bool,
            start_inputs=start_inputs, end_fn=end_fn,
            next_inputs_fn=next_inputs_fn)
        my_decoder = seq2seq.BasicDecoder(
            cell=cell,
            helper=helper,
            initial_state=cell.zero_state(
                dtype=tf.float32, batch_size=batch_size))
        output_size = my_decoder.output_size
        output_dtype = my_decoder.output_dtype
        self.assertEqual(
            seq2seq.BasicDecoderOutput(cell_depth, cell_depth),
            output_size)
        self.assertEqual(
            seq2seq.BasicDecoderOutput(tf.float32, tf.bool),
            output_dtype)

        (first_finished, first_inputs, first_state) = my_decoder.initialize()
        (step_outputs, step_state, step_next_inputs,
         step_finished) = my_decoder.step(
             tf.constant(0), first_inputs, first_state)
        batch_size_t = my_decoder.batch_size

        self.assertIsInstance(first_state, tf.nn.rnn_cell.LSTMStateTuple)
        self.assertIsInstance(step_state, tf.nn.rnn_cell.LSTMStateTuple)
        self.assertIsInstance(step_outputs, seq2seq.BasicDecoderOutput)
        self.assertEqual((batch_size, cell_depth), step_outputs[0].get_shape())
        self.assertEqual((batch_size, cell_depth), step_outputs[1].get_shape())
        self.assertEqual((batch_size, cell_depth), first_state[0].get_shape())
        self.assertEqual((batch_size, cell_depth), first_state[1].get_shape())
        self.assertEqual((batch_size, cell_depth), step_state[0].get_shape())
        self.assertEqual((batch_size, cell_depth), step_state[1].get_shape())

        sess.run(tf.global_variables_initializer())
        sess_results = sess.run({
            "batch_size": batch_size_t,
            "first_finished": first_finished,
            "first_inputs": first_inputs,
            "first_state": first_state,
            "step_outputs": step_outputs,
            "step_state": step_state,
            "step_next_inputs": step_next_inputs,
            "step_finished": step_finished
        })

        sample_ids = sess_results["step_outputs"].sample_id
        self.assertEqual(output_dtype.sample_id, sample_ids.dtype)
        expected_step_finished = sample_ids[:, end_token]
        expected_step_next_inputs = sample_ids.astype(np.float32)
        self.assertAllEqual(expected_step_finished,
                            sess_results["step_finished"])
        self.assertAllEqual(expected_step_next_inputs,
                            sess_results["step_next_inputs"])

if __name__ == "__main__":
  tf.test.main()
