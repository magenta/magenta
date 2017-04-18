import tensorflow as tf
import numpy as np
import copy
from common.sampling.forward import ForwardSample, batchify_dict

class ParticleFilter(ForwardSample):

	def __init__(self, model, checkpoint_dir, batch_size=1):
		super(ParticleFilter, self).__init__(model, checkpoint_dir, batch_size)

		# Particle filtering stuff
		self.log_probabilities = np.zeros(self.batch_size)
		self.sample_placeholder = tf.placeholder(dtype = tf.int32, shape=[batch_size, self.model.timeslice_size], name = "samples")
		self.log_probability_node = self.dist.log_prob(self.sample_placeholder)

	"""
	Generate for one time step
	Returns next rnn state as well as the sampled time slice
	"""
	def sample_step(self, rnn_state, rnn_input, condition_dict):
		# First, we run the graph to get the rnn outputs and next state
		feed_dict = { self.input_placeholder: rnn_input, self.rnn_state: rnn_state }
		next_state, outputs = self.sess.run([self.final_state, self.rnn_outputs], feed_dict)

		# Next, we slice out the last timeslice of the outputs--we only want to
		#    compute a distribution over that
		# (Can't do this in the graph b/c we don't know how long initial_timeslices will be up-front)
		seq_len = outputs.shape[1]
		if seq_len > 1:
			# slices out the last time entry but keeps the tensor 3D
			outputs = outputs[:, seq_len-1, np.newaxis, :]

		# Then, we feed this into the rest of the graph to sample from the
		#    timeslice distribution
		feed_dict = {self.condition_dict_placeholders[name]: condition_dict[name] for name in condition_dict}
		feed_dict[self.rnn_outputs] = outputs
		sample = self.sess.run(self.sampled_timeslice, feed_dict)
		matching = False

		if condition_dict:
			# Keep resampling until there are some samples that satisfy the conditions specified.
			while not matching:
				feed_dict[self.sample_placeholder] = sample 
				log_probabilities = np.zeros((self.batch_size,))
				for i in range(self.batch_size):
					log_probabilities[i] += self.model.eval_factor_function(sample[i], condition_dict['known_notes'][i][0])
				log_probabilities = np.exp(log_probabilities)
				if sum(log_probabilities) > 0:
					matching = True
				else:
					sample = self.sess.run(self.sampled_timeslice, feed_dict)

			normalized_log_probabilities = np.array([float(i/sum(log_probabilities)) for i in log_probabilities])
			new_sample = np.zeros(sample.shape)
			
			# Resample from the distribution which favors samples that satisfy the conditions specified.
			for i in range(self.batch_size):
				new_dist = np.random.multinomial(1, normalized_log_probabilities)
				new_sample[i] = np.matmul(new_dist.reshape(1, -1), sample)

			sample = new_sample
		
		# Keep track of the log probability.
		feed_dict[self.sample_placeholder] = sample
		log_probabilities = self.sess.run(self.log_probability_node, feed_dict)
		log_probabilities = np.sum(log_probabilities, axis = 1)
		self.log_probabilities += log_probabilities

		# Finally, we reshape the sample to be 3D again (the Distribution is over 2D [batch, depth]
		#    tensors--we need to reshape it to [batch, time, depth], where time=1)
		sample = sample[:,np.newaxis,:]

		return next_state, sample



