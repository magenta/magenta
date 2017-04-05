import tensorflow as tf
from common.models.sequenceGenerativeModel import SequenceGenerativeModel
from common.distributions.Nade import NADE

"""
An RNN that models a sequence of NADE objects, where the parameters of NADE
   are determined by the RNN hidden state.
"""
class RNNNade(SequenceGenerativeModel):

	def __init__(self, hparams, sequence_encoder):
		super(RNNNade, self).__init__(hparams, sequence_encoder)

	def get_step_dist(self, rnn_outputs, condition_dict):
		# Combine batch and time dimensions so we have a 2D tensor (i.e. a list of
		#    of opts.num_notes-long tensors). Need for layers.linear, I think?
		outputs_flat = tf.reshape(rnn_outputs, [-1, self.rnn_cell().output_size])
		# Compute logits for sigmoid cross entropy loss
		#logits_flat = tf.contrib.layers.linear(outputs_flat, self.timeslice_size)
		# Create and return NADE dist
		# (sample dtype is float so samples can be fed right back into inputs)
		dist = NADE(#logits=logits_flat, dtype=tf.float32)
		return dist
