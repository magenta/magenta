import tensorflow as tf

# Load a padded batch of data from the training data file
def loadBatch(opts):
	filename = opts['filename']
	batchSize = opts['batch_size']
	numThreads = opts['num_threads']
	filenameQueue = tf.train.string_input_producer([filename])
	reader = tf.TFRecordReader()
	_, serializedExample = reader.read(filenameQueue)
	_, sequenceFeatures = tf.parse_single_sequence_example(
		serialized = serializedExample,
		sequence_features = {
			'inputs': tf.FixedLenSequenceFeature([54], dtype=tf.float32),
			'outputs': tf.FixedLenSequenceFeature([54], dtype=tf.float32)
		}
	)
	inputs = sequenceFeatures['inputs']
	outputs = sequenceFeatures['outputs']
	lengths = tf.shape(inputs)[0]

	return tf.train.batch(
		tensors=[inputs, outputs, lengths],
		batch_size=batchSize,
		num_threads=numThreads,
		capacity=3*batchSize,
		dynamic_pad=True
	)


# Stolen from magenta/models/shared/events_rnn_graph
def make_rnn_cell(rnn_layer_sizes,
				  dropout_keep_prob=1.0,
				  attn_length=0,
				  base_cell=tf.nn.rnn_cell.BasicLSTMCell,
				  state_is_tuple=True):
  cells = []
  for num_units in rnn_layer_sizes:
	cell = base_cell(num_units, state_is_tuple=state_is_tuple)
	cell = tf.nn.rnn_cell.DropoutWrapper(
		cell, output_keep_prob=dropout_keep_prob)
	cells.append(cell)

  cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
  if attn_length:
	cell = tf.contrib.rnn.AttentionCellWrapper(
		cell, attn_length, state_is_tuple=state_is_tuple)

  return cell


# Build the graph which evaluates training loss
def trainingLoss(inputs, targets, lengths, opts):
	cell = make_rnn_cell(opts['rnn_layer_sizes'],
						 dropout_keep_prob=opts['dropout_keep_prob'],
						 attn_length=opts['attn_length'])

	# Initialize hidden state to zero
	initial_state = cell.zero_state(opts['batch_size'], tf.float32)

	# Run the RNN
	outputs, final_state = tf.nn.dynamic_rnn(
		cell, inputs, initial_state=initial_state, parallel_iterations=1,
		swap_memory=True)

	# Combine batch and time dimensions so we have a 2D tensor (i.e. a list of
	#    of opts.num_notes-long tensors). Need for layers.linear, I think?
	outputs_flat = tf.reshape(outputs, [-1, cell.output_size])
	targets_flat = tf.reshape(targets, [-1, opts['num_notes']])
	# Compute logits for sigmoid cross entropy loss
	logits_flat = tf.contrib.layers.linear(outputs_flat, opts['num_notes'])
	num_time_slices = tf.to_float(tf.reduce_sum(lengths))
	# Mask out the stuff that was past the end of each training sequence (due to padding)
	mask_flat = tf.reshape(tf.sequence_mask(lengths, dtype=tf.float32), [-1])
	# Compute sigmoid cross entropy loss
	sce = tf.nn.sigmoid_cross_entropy_with_logits(logits_flat, targets_flat)
	# Sum across 'space' (i.e. entries in a single time slice), then across time+batch
	sce = tf.reduce_sum(sce, 1)
	loss = tf.reduce_sum(mask_flat * sce) / num_time_slices

	return loss


# Build the graph that actually does an iteration of training
def trainingIteration(loss, opts):
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(
			opts['initial_learning_rate'], global_step, opts['decay_steps'],
			opts['decay_rate'], staircase=True)
	opt = tf.train.AdamOptimizer(learning_rate)
	params = tf.trainable_variables()
	gradients = tf.gradients(loss, params)
	clipped_gradients, _ = tf.clip_by_global_norm(gradients,
												  opts['gradient_clip_norm'])
	train_op = opt.apply_gradients(zip(clipped_gradients, params),
								   global_step)

	return train_op, learning_rate, global_step

# Assemble the complete graph and do training
def doTraining(opts):
	inputs, targets, lengths = loadBatch(opts)
	loss = trainingLoss(inputs, targets, lengths, opts)
	train_op, learning_rate, global_step = trainingIteration(loss, opts)

	sv = tf.train.Supervisor(logdir=opts['log_dir'], save_model_secs=opts['save_model_secs'],
						   global_step=global_step)

	tf.logging.set_verbosity(tf.logging.INFO)
	with sv.managed_session() as sess:
		global_step_ = sess.run(global_step)
		# if opts.n_training_iters and global_step_ >= opts.n_training_iters:
		# 	tf.logging.info('This checkpoint\'s global_step value is already %d, '
		# 				  'which is greater or equal to the specified '
		# 				  'num_training_steps value of %d. Exiting training.',
		# 				  global_step_, num_training_steps)
		# 	return
		tf.logging.info('Starting training loop...')
		while global_step_ < opts['n_training_iters']:
			if sv.should_stop():
				break
			if (global_step_ + 1) % opts['summary_frequency'] == 0:
				(global_step_, learning_rate_, loss_, _) = \
					sess.run([global_step, learning_rate, loss, train_op])
				tf.logging.info('Global Step: %d - '
								'Learning Rate: %.5f - '
								'Loss: %.3f - ',
								global_step_, learning_rate_, loss_)
			else:
				global_step_, _ = sess.run([global_step, train_op])
		sv.saver.save(sess, sv.save_path, global_step=sv.global_step)
		tf.logging.info('Training complete.')

# ---------------------------------------------------------------------------------------

opts = {
	# Data prep
	'filename': 'data/jsb_chorales_train.tfrecord',
	'num_threads': 2,
	'batch_size': 128,

	# Model
	'num_notes': 54,
	'rnn_layer_sizes': [128],
	'dropout_keep_prob': 1.0,
	'attn_length': None,

	# Training
	'n_training_iters': 20000,
	'initial_learning_rate': 0.001,
	'decay_steps': 1000,
	'decay_rate': 0.95,
	'gradient_clip_norm': 5,
	'summary_frequency': 10,
	'log_dir': './log',
	'save_model_secs': 30
}

doTraining(opts)



