import tensorflow as tf
import numpy as np

# TODO: 'temperature' parameter to control how random the sampling is?

"""
Turn a 1-D array into a 3D array (shape is [batch, time, depth], where time == 1)
"""
def batchify_array(x, batch_size):
	return np.tile(x, (batch_size, 1, 1))

def batchify_dict(dic, batch_size):
	return { name: batchify_array(x, batch_size) for name,x in dic.iteritems() }

def concatenate_dicts(dicts, axis):
	keys = dicts[0].keys()
	return { name: np.concatenate([dic[name] for dic in dicts], axis) for name in keys }

class ForwardSample(object):

	def __init__(self, model, checkpoint_dir, batch_size=1):
		self.model = model
		self.batch_size = batch_size

		# Construct a graph that takes placeholder inputs and produces the time slice Distribution
		self.input_dict_placeholders = {
			name: tf.placeholder(dtype=tf.float32, shape=[batch_size,None]+shape) for name,shape in model.input_shapes.iteritems()
		}
		self.condition_dict_placeholders = {
			name: tf.placeholder(dtype=tf.float32, shape=[batch_size,None]+shape) for name,shape in model.condition_shapes.iteritems()
		}
		# self.rnn_state = model.initial_state(1)
		self.rnn_state = model.initial_state(batch_size)
		self.final_state, self.rnn_outputs = model.run_rnn(self.rnn_state, self.input_dict_placeholders)
		self.dist = model.get_step_dist(self.rnn_outputs, self.condition_dict_placeholders)
		self.sampled_timeslice = self.dist.sample()

		# # Op for initializing rnn state
		# self.initial_state_op = model.initial_state(1)

		# Setup a session and restore saved variables
		self.sess = tf.Session()
		checkpoint_filename = tf.train.latest_checkpoint(checkpoint_dir)
		saver = tf.train.Saver()
		saver.restore(self.sess, checkpoint_filename)

	"""
	Draw forward samples from a SequenceGenerativeModel for n_steps.
	initial_input_dicts: a sequence of input dictionaries used to 'prime' the model.
	   If None, then use the model's default initial input dict.
	condition_dicts: a sequence of input dictionaries that provide additional conditioning
	   information as sampling is happening.
	returns: a list of list of timeslice samples from the model
	"""
	def sample(self, n_steps, initial_input_dicts=None, condition_dicts=None):
		
		input_dicts = None
		if initial_input_dicts is not None:
			# Batchify all the input dicts
			initial_input_dicts = [ batchify_dict(dic, self.batch_size) for dic in initial_input_dicts ]
			# Join all of these along the time dimension
			inputs_dicts = [ concatenate_dicts(initial_input_dicts, 1) ]
		else:
			# Use the model's default initial input
			input_dicts = [ batchify_dict(self.model.default_initial_input_dict(), self.batch_size) ]

		if condition_dicts is not None:
			# Batchify these, too
			condition_dicts = [ batchify_dict(dic, self.batch_size) for dic in condition_dicts ]

		# Initialize state
		rnn_state = self.sess.run(self.rnn_state)

		# Iteratively draw sample, and feed the sample in to the next input dict
		samples = []
		for i in range(n_steps):
			input_dict = input_dicts[i]
			condition_dict = {} if (condition_dicts is None) else condition_dicts[i]
			rnn_state, sample = self.sample_step(rnn_state, input_dict, condition_dict)
			samples.append(sample)
			input_dicts.append(self.model.sample_to_next_input_dict(sample, input_dicts))

		# Concatenate samples along time dimension to make one big block.
		# Then split along batch dimension, and then again along time dimension, so return value
		#    is a triply-nested list
		sampleBlock = np.concatenate(samples, 1)
		seqs = [seq[0] for seq in np.split(sampleBlock, self.batch_size, axis=0)]
		return [[slic[0] for slic in np.split(seq, n_steps, axis=0)] for seq in seqs]

	"""
	Generate for one time step
	Returns next rnn state as well as the sampled time slice
	"""
	def sample_step(self, rnn_state, input_dict, condition_dict):
		# First, we run the graph to get the rnn outputs and next state
		feed_dict = {self.input_dict_placeholders[name]: input_dict[name] for name in input_dict.keys() }
		feed_dict[self.rnn_state] = rnn_state
		next_state, outputs = self.sess.run([self.final_state, self.rnn_outputs], feed_dict)

		# Next, we slice out the last timeslice of the outputs--we only want to
		#    compute a distribution over that
		# (Can't do this in the graph b/c we don't know how long initial_input_dicts will be up-front)
		seq_len = outputs.shape[1]
		if seq_len > 1:
			# slices out the last time entry but keeps the tensor 3D
			outputs = outputs[:, seq_len-1, np.newaxis, :]

		# Then, we feed this into the rest of the graph to sample from the
		#    timeslice distribution
		feed_dict = {self.condition_dict_placeholders[name]: condition_dict[name] for name in condition_dict}
		feed_dict[self.rnn_outputs] = outputs
		sample = self.sess.run(self.sampled_timeslice, feed_dict)

		# Finally, we reshape the sample to be 3D again (the Distribution is over 2D [batch, depth]
		#    tensors--we need to reshape it to [batch, time, depth], where time=1)
		sample = sample[:,np.newaxis,:]

		return next_state, sample



