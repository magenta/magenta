from common.models.sequenceModel import SequenceModel
from common.models.utils import make_rnn_cell
import tensorflow as tf

"""
An RNN that models a sequence of binary random vectors, where the components of each
   vector are modeled independently from one other given the RNN hidden state.
"""
class RNNIndependent(SequenceModel):

	def __init__(self, hparams):
		self.hparams = hparams

	def training_loss(self, batch):
		inputs = batch['inputs']
		targets = batch['outputs']
		lengths = batch['lengths']

		batch_size = tf.shape(inputs)[0]
		timeslice_size = self.hparams.timeslice_size

		cell = make_rnn_cell(self.hparams.rnn_layer_sizes,
							 dropout_keep_prob=self.hparams.dropout_keep_prob,
							 attn_length=self.hparams.attn_length)

		# Initialize hidden state to zero
		initial_state = cell.zero_state(batch_size, tf.float32)

		# Run the RNN
		outputs, final_state = tf.nn.dynamic_rnn(
			cell, inputs, initial_state=initial_state, parallel_iterations=1,
			swap_memory=True)

		# Combine batch and time dimensions so we have a 2D tensor (i.e. a list of
		#    of opts.num_notes-long tensors). Need for layers.linear, I think?
		outputs_flat = tf.reshape(outputs, [-1, cell.output_size])
		targets_flat = tf.reshape(targets, [-1, timeslice_size])
		# Compute logits for sigmoid cross entropy loss
		logits_flat = tf.contrib.layers.linear(outputs_flat, timeslice_size)
		num_time_slices = tf.to_float(tf.reduce_sum(lengths))
		# Mask out the stuff that was past the end of each training sequence (due to padding)
		mask_flat = tf.reshape(tf.sequence_mask(lengths, dtype=tf.float32), [-1])
		# Compute sigmoid cross entropy loss
		sce = tf.nn.sigmoid_cross_entropy_with_logits(logits_flat, targets_flat)
		# Sum across 'space' (i.e. entries in a single time slice), then across time+batch
		sce = tf.reduce_sum(sce, 1)
		loss = tf.reduce_sum(mask_flat * sce) / num_time_slices

		return loss

	# TODO: What does this class need to provide to support forward sampling, beam search, MCMC??