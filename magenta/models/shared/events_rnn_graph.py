# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provides function to build an event sequence RNN model's graph."""

# internal imports
import tensorflow as tf
import magenta


def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=tf.contrib.rnn.BasicLSTMCell,
                  state_is_tuple=False):
  """Makes a RNN cell from the given hyperparameters.

  Args:
    rnn_layer_sizes: A list of integer sizes (in units) for each layer of the
        RNN.
    dropout_keep_prob: The float probability to keep the output of any given
        sub-cell.
    attn_length: The size of the attention vector.
    base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.
    state_is_tuple: A boolean specifying whether to use tuple of hidden matrix
        and cell matrix as a state instead of a concatenated matrix.

  Returns:
      A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
  """
  cells = []
  for num_units in rnn_layer_sizes:
    cell = base_cell(num_units, state_is_tuple=state_is_tuple)
    cell = tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob=dropout_keep_prob)
    cells.append(cell)

  cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
  if attn_length:
    cell = tf.contrib.rnn.AttentionCellWrapper(
        cell, attn_length, state_is_tuple=state_is_tuple)

  return cell

def dilated_cnn(inputs,
                 initial_state,
                 input_size,
                 block_num=1,
                 block_size=7,
                 dropout_keep_prob=1.0,
                 residual_cnl=16,
                 dilation_cnl=8,
                 output_cnl=32,
                 use_gate=True,
                 use_step=True,
                 mode='train'):
  """
  Returns outputs of dilated CNN from the given hyperparameters.
  This model uses convolution neural network as generative model like Wavenet.

  Args:
    inputs: Model inputs.
    initial_state: A numpy array containing initial padding buffer of
        generative CNN model.
    input_size: The size of input vector.
    block_num: The number of dilated convolution blocks.
    block_size: The size of dilated convolution blocks.
    dropout_keep_prob: The float probability to keep the output of any given
        sub-cell.
    residual_cnl: The size of hidden residual state.
    dilation_cnl: The size of hidden dilation state.
    output_cnl: The size of output vector.
    use_gate: A boolean specifying whether to use gated activation units.
    use_step: A boolean specifying whether to use skip connection.
    mode: 'train', 'eval', or 'generate'.

  Returns:
    outputs: Model outputs
    final_state: A numpy array containing next padding buffer of generative
        CNN model.
  """
  if mode == 'train':
    is_training = True
  else:
    is_training = False

  dilation = [2**i for i in range(block_size)]*block_num
  batch_num = tf.shape(inputs)[0]
  h = tf.reshape(inputs, [batch_num,-1,1,input_size])
  dlt_sum = [sum(dilation[:i]) for i in range(len(dilation))]
  dlt_sum.append(sum(dilation))

  with tf.variable_scope("first_conv"):
    h = tf.contrib.layers.batch_norm(h, decay=0.999, center=True, scale=True,
                              updates_collections=None, is_training=is_training,
                              scope="first_conv", reuse=True)
    first_weights = tf.get_variable(
        "first_weights", [1,1,input_size,residual_cnl],
        initializer=tf.random_normal_initializer())
    h = tf.nn.conv2d(h, first_weights, strides=[1,1,1,1], padding='SAME')
  final_state = []
  if use_step:
    step = []
  for i,dlt in enumerate(dilation):
    pad = initial_state[:,dlt_sum[i]*residual_cnl:dlt_sum[i+1]*residual_cnl]
    pad = tf.reshape(pad,[batch_num,dlt,1,residual_cnl])
    _h = h
    h = tf.concat([pad,h],1)
    _fs = tf.reshape(h[:,-dlt:,:,:],[batch_num,dlt*residual_cnl])
    final_state.append(_fs)

    with tf.variable_scope("conv{}".format(i)):
      if use_gate:
        gate_weights = tf.get_variable(
            "gate_weights", [2,1,residual_cnl,dilation_cnl],
            initializer=tf.random_normal_initializer())
        gate_biases = tf.get_variable(
            "gate_biases", [dilation_cnl],
            initializer=tf.constant_initializer(0.0))
        gate = tf.nn.atrous_conv2d(
            h, gate_weights, dlt, padding="VALID")
        gate = tf.contrib.layers.batch_norm(gate, decay=0.999, center=True,
                                        scale=True, updates_collections=None,
                                        is_training=is_training,
                                        scope="gate_bn{}".format(i), reuse=True)
        gate = tf.sigmoid(tf.nn.bias_add(gate, gate_biases))

      filter_weights = tf.get_variable(
          "filter_weights", [2,1,residual_cnl,dilation_cnl],
          initializer=tf.random_normal_initializer())
      filter_biases = tf.get_variable(
          "filter_biases", [dilation_cnl],
          initializer=tf.constant_initializer(0.0))
      filtr = tf.nn.atrous_conv2d(
          h, filter_weights, dlt, padding="VALID")
      filtr = tf.contrib.layers.batch_norm(filtr, decay=0.999, center=True,
                                    scale=True, updates_collections=None,
                                    is_training=is_training,
                                    scope="filter_bn{}".format(i), reuse=True)
      filtr = tf.tanh(tf.nn.bias_add(filtr, filter_biases))

      after_weights = tf.get_variable(
          "after_weights", [1,1,dilation_cnl,residual_cnl],
          initializer=tf.random_normal_initializer())
      after_biases = tf.get_variable(
          "after_biases", [residual_cnl],
          initializer=tf.constant_initializer(0.0))
      if use_gate:
        after = tf.nn.conv2d(
            gate*filtr, after_weights,strides=[1,1,1,1],padding='SAME')
      else:
        after = tf.nn.conv2d(
            filtr, after_weights,strides=[1,1,1,1],padding='SAME')
        after = tf.nn.bias_add(after, after_biases)

      if use_step:
        step.append(after)
        h = after + _h

  if use_step:
    step = tf.concat(step,3)
    step_weights = tf.get_variable(
        "step_weights", [1,1,residual_cnl*len(dilation),output_cnl],
        initializer=tf.random_normal_initializer())
    h = tf.nn.conv2d(step, step_weights, strides=[1,1,1,1], padding='SAME')
    h = tf.contrib.layers.batch_norm(h, decay=0.999, center=True, scale=True,
                              updates_collections=None, is_training=is_training,
                              scope="step_bn", reuse=True)
    h = tf.nn.relu(h)

    last_weights = tf.get_variable(
        "last_weights", [1,1,output_cnl,output_cnl],
        initializer=tf.random_normal_initializer())
  else:
    last_weights = tf.get_variable(
        "last_weights", [1,1,residual_cnl,output_cnl],
        initializer=tf.random_normal_initializer())

  last_biases = tf.get_variable(
      "last_biases", [output_cnl], initializer=tf.constant_initializer(0.0))
  h = tf.nn.conv2d(h, last_weights, strides=[1,1,1,1], padding='SAME')
  h = tf.contrib.layers.batch_norm(h, decay=0.999, center=True, scale=True,
                            updates_collections=None, is_training=is_training,
                            scope="last_bn", reuse=True)
  h = tf.nn.relu(tf.nn.bias_add(h, last_biases))

  h = tf.nn.dropout(h, dropout_keep_prob)
  final_state = tf.concat(final_state,1)
  outputs = tf.reshape(h, [batch_num,-1,output_cnl])
  return outputs,final_state

def get_dilated_cnn_initial_state(batch_size,
                                  dtype,
                                  input_size,
                                  block_num=1,
                                  block_size=7,
                                  residual_cnl=16):
  """
  Returns a initial_state using in the dilated_cnn method.

  Args:
    batch_size: A size of input batch.
    dtype: A type of initial_state
    input_size: The size of input vector.
    block_num: The number of dilated convolution blocks.
    block_size: The size of dilated convolution blocks.
    residual_cnl: The size of hidden residual state.
  Returns:
    initial_state: A numpy array containing initial_state of dilated_cnn.
  """
  dilation = [2**i for i in range(block_size)]*block_num
  initial_state = tf.zeros([batch_size,sum(dilation)*residual_cnl])
  return initial_state

def build_graph(mode, config, sequence_example_file_paths=None):
  """Builds the TensorFlow graph.

  Args:
    mode: 'train', 'eval', or 'generate'. Only mode related ops are added to
        the graph.
    config: An EventSequenceRnnConfig containing the encoder/decoder and HParams
        to use.
    sequence_example_file_paths: A list of paths to TFRecord files containing
        tf.train.SequenceExample protos. Only needed for training and
        evaluation. May be a sharded file of the form.

  Returns:
    A tf.Graph instance which contains the TF ops.

  Raises:
    ValueError: If mode is not 'train', 'eval', or 'generate'.
  """
  if mode not in ('train', 'eval', 'generate'):
    raise ValueError("The mode parameter must be 'train', 'eval', "
                     "or 'generate'. The mode parameter was: %s" % mode)

  hparams = config.hparams
  encoder_decoder = config.encoder_decoder

  tf.logging.info('hparams = %s', hparams.values())

  input_size = encoder_decoder.input_size
  num_classes = encoder_decoder.num_classes
  no_event_label = encoder_decoder.default_event_label

  with tf.Graph().as_default() as graph:
    inputs, labels, lengths, = None, None, None
    state_is_tuple = True

    if mode == 'train' or mode == 'eval':
      inputs, labels, lengths = magenta.common.get_padded_batch(
          sequence_example_file_paths, hparams.batch_size, input_size)

    elif mode == 'generate':
      inputs = tf.placeholder(tf.float32, [hparams.batch_size, None,
                                           input_size])
      # If state_is_tuple is True, the output RNN cell state will be a tuple
      # instead of a tensor. During training and evaluation this improves
      # performance. However, during generation, the RNN cell state is fed
      # back into the graph with a feed dict. Feed dicts require passed in
      # values to be tensors and not tuples, so state_is_tuple is set to False.
      state_is_tuple = False

    if hparams.dilated_cnn:
        initial_state = get_dilated_cnn_initial_state(
            hparams.batch_size, tf.float32, input_size,
            block_num=hparams.block_num,block_size=hparams.block_size,
            residual_cnl=hparams.residual_cnl)
        outputs, final_state = dilated_cnn(
            inputs, initial_state, input_size,block_num=hparams.block_num,
            block_size=hparams.block_size,
            dropout_keep_prob=hparams.dropout_keep_prob,
            residual_cnl=hparams.residual_cnl,
            dilation_cnl=hparams.dilation_cnl,
            output_cnl=hparams.output_cnl,
            use_gate=hparams.use_gate, use_step=hparams.use_step,
            mode=mode)

        outputs_flat = tf.reshape(outputs, [-1, hparams.output_cnl])
    else:
        cell = make_rnn_cell(hparams.rnn_layer_sizes,
                             dropout_keep_prob=hparams.dropout_keep_prob,
                             attn_length=hparams.attn_length,
                             state_is_tuple=state_is_tuple)

        initial_state = cell.zero_state(hparams.batch_size, tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(
            cell, inputs, initial_state=initial_state, parallel_iterations=1,
            swap_memory=True)

        outputs_flat = tf.reshape(outputs, [-1, cell.output_size])
    logits_flat = tf.contrib.layers.linear(outputs_flat, num_classes)

    if mode == 'train' or mode == 'eval':
      labels_flat = tf.reshape(labels, [-1])
      mask = tf.sequence_mask(lengths)
      if hparams.skip_first_n_losses:
        skip = tf.minimum(lengths, hparams.skip_first_n_losses)
        skip_mask = tf.sequence_mask(skip, maxlen=tf.reduce_max(lengths))
        mask = tf.logical_and(mask, tf.logical_not(skip_mask))
      mask = tf.cast(mask, tf.float32)
      mask_flat = tf.reshape(mask, [-1])

      num_logits = tf.to_float(tf.reduce_sum(lengths))

      with tf.control_dependencies(
          [tf.Assert(tf.greater(num_logits, 0.), [num_logits])]):
        softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_flat, logits=logits_flat)
      loss = tf.reduce_sum(mask_flat * softmax_cross_entropy) / num_logits
      perplexity = (tf.reduce_sum(mask_flat * tf.exp(softmax_cross_entropy)) /
                    num_logits)

      correct_predictions = tf.to_float(
          tf.nn.in_top_k(logits_flat, labels_flat, 1)) * mask_flat
      accuracy = tf.reduce_sum(correct_predictions) / num_logits * 100

      event_positions = (
          tf.to_float(tf.not_equal(labels_flat, no_event_label)) * mask_flat)
      event_accuracy = (
          tf.reduce_sum(tf.multiply(correct_predictions, event_positions)) /
          tf.reduce_sum(event_positions) * 100)

      no_event_positions = (
          tf.to_float(tf.equal(labels_flat, no_event_label)) * mask_flat)
      no_event_accuracy = (
          tf.reduce_sum(tf.multiply(correct_predictions, no_event_positions)) /
          tf.reduce_sum(no_event_positions) * 100)

      global_step = tf.Variable(0, trainable=False, name='global_step')

      tf.add_to_collection('loss', loss)
      tf.add_to_collection('perplexity', perplexity)
      tf.add_to_collection('accuracy', accuracy)
      tf.add_to_collection('global_step', global_step)

      summaries = [
          tf.summary.scalar('loss', loss),
          tf.summary.scalar('perplexity', perplexity),
          tf.summary.scalar('accuracy', accuracy),
          tf.summary.scalar(
              'event_accuracy', event_accuracy),
          tf.summary.scalar(
              'no_event_accuracy', no_event_accuracy),
      ]

      if mode == 'train':
        learning_rate = tf.train.exponential_decay(
            hparams.initial_learning_rate, global_step, hparams.decay_steps,
            hparams.decay_rate, staircase=True, name='learning_rate')

        opt = tf.train.AdamOptimizer(learning_rate)
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                      hparams.clip_norm)
        train_op = opt.apply_gradients(zip(clipped_gradients, params),
                                       global_step)
        tf.add_to_collection('learning_rate', learning_rate)
        tf.add_to_collection('train_op', train_op)

        summaries.append(tf.summary.scalar(
            'learning_rate', learning_rate))

      if mode == 'eval':
        summary_op = tf.summary.merge(summaries)
        tf.add_to_collection('summary_op', summary_op)

    elif mode == 'generate':
      temperature = tf.placeholder(tf.float32, [])
      softmax_flat = tf.nn.softmax(
          tf.div(logits_flat, tf.fill([num_classes], temperature)))
      softmax = tf.reshape(softmax_flat, [hparams.batch_size, -1, num_classes])

      tf.add_to_collection('inputs', inputs)
      tf.add_to_collection('initial_state', initial_state)
      tf.add_to_collection('final_state', final_state)
      tf.add_to_collection('temperature', temperature)
      tf.add_to_collection('softmax', softmax)

  return graph
