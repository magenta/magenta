import tensorflow as tf

"""
Build the graph that does an iteration of training
"""
def training_iter(loss, hparams):
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(
			hparams.initial_learning_rate, global_step, hparams.decay_steps,
			hparams.decay_rate, staircase=True)
	opt = tf.train.AdamOptimizer(learning_rate)
	params = tf.trainable_variables()
	gradients = tf.gradients(loss, params)
	clipped_gradients, _ = tf.clip_by_global_norm(gradients,
												  hparams.gradient_clip_norm)
	train_op = opt.apply_gradients(zip(clipped_gradients, params),
								   global_step)

	return train_op, learning_rate, global_step

"""
Run training on a model for some number of iterations
"""
def train(model, dataset, hparams):
	batch = dataset.load_batch(hparams.batch_size, hparams.num_threads)
	loss = model.training_loss(batch)
	train_op, learning_rate, global_step = training_iter(loss, hparams)

	sv = tf.train.Supervisor(logdir=hparams.log_dir, save_model_secs=hparams.save_model_secs,
						   global_step=global_step)

	tf.logging.set_verbosity(tf.logging.INFO)
	with sv.managed_session() as sess:
		global_step_ = sess.run(global_step)
		# if hparams..n_training_iters and global_step_ >= hparams..n_training_iters:
		# 	tf.logging.info('This checkpoint\'s global_step value is already %d, '
		# 				  'which is greater or equal to the specified '
		# 				  'num_training_steps value of %d. Exiting training.',
		# 				  global_step_, num_training_steps)
		# 	return
		tf.logging.info('Starting training loop...')
		while global_step_ < hparams.n_training_iters:
			if sv.should_stop():
				break
			if (global_step_ + 1) % hparams.summary_frequency == 0:
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