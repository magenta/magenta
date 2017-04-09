import tensorflow as tf
from common.models.sequenceGenerativeModel import SequenceGenerativeModel
from common.distributions import Nade

"""
An RNN that models a sequence of NADE objects, where the parameters of each NADE
   are determined by the RNN hidden state.
"""
class RNNNade(SequenceGenerativeModel):

	def __init__(self, hparams, sequence_encoder,size_hidden_layer=50):
		super(RNNNade, self).__init__(hparams, sequence_encoder)
		self.size_hidden_layer=size_hidden_layer

	def get_step_dist(self, rnn_outputs, condition_dict):
		with tf.variable_scope('NADE_model') as scope:
			W = tf.get_variable("W", shape = (self.timeslice_size, self.size_hidden_layer), initializer = tf.contrib.layers.xavier_initializer())
			V = tf.get_variable("V", shape = (self.size_hidden_layer, self.timeslice_size), initializer = tf.contrib.layers.xavier_initializer())
			scope.reuse_variables()

		# Combine batch and time dimensions so we have a 2D tensor (i.e. a list of
		#    of opts.num_notes-long tensors). Need for layers.linear, I think?
		outputs_flat = tf.reshape(rnn_outputs, [-1, self.rnn_cell().output_size])
		# Compute parameters a and b of NADE object
		b = tf.contrib.layers.fully_connected(inputs = output_flat, num_outputs = self.timeslice_size, activation_fn = None,
			weights_initializer = tf.contrib.layers.xavier_initializer(), biases_initializer = tf.contrib.layers.xavier_initializer(),
			reuse = True, trainable = True, scope = 'NADE_model/b')
		a = tf.contrib.layers.fully_connected(inputs = output_flat, num_outputs = size_hidden_layer, activation_fn = None,
			weights_initializer = tf.contrib.layers.xavier_initializer(), biases_initializer = tf.contrib.layers.xavier_initializer(),
			reuse = True, trainable = True, scope='NADE_model/a')

		# Create and return NADE object
		# (sample dtype is float so samples can be fed right back into inputs)
		dist = NADE(a,b,W,V,dtype=tf.float32)
		return dist
