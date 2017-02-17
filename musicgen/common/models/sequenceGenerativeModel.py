import abc
from common.models.model import Model
from common.models.utils import make_rnn_cell
import tensorflow as tf
import numpy as np

"""
Abstract base class for generative sequence models
"""
class SequenceGenerativeModel(Model):

	__metaclass__ = abc.ABCMeta

	def __init__(self, hparams):
		super(SequenceGenerativeModel, self).__init__(hparams)
		self._rnn_cell = None

	@property
  	def timeslice_size(self):
		return self.hparams.timeslice_size

	"""
	Names and shapes of all the inputs this model expects in its input dicts
	"""
	@property
	def input_shapes(self):
		return {'inputs': [self.timeslice_size]}

	"""
	Names and shapes of all the conditioning data this model expects in its condition dicts
	"""
	@property
	def condition_shapes(self):
		return {}

	"""
	Build the sub-graph for the RNN cell
	Result is cached, so the same sub-graph can be re-used(?)
	"""
	def rnn_cell(self):
		if self._rnn_cell is None:
			self._rnn_cell = make_rnn_cell(self.hparams.rnn_layer_sizes,
							 	dropout_keep_prob=self.hparams.dropout_keep_prob,
							 	attn_length=self.hparams.attn_length)
		return self._rnn_cell

	"""
	Get an RNN initial state vector for a given batch size
	"""
	def initial_state(self, batch_size):
		return self.rnn_cell().zero_state(batch_size, tf.float32)

	"""
	Initial input dict to use for this model in the absence of any priming inputs.
	By default, this just uses a zero vector of size self.timeslice_size.
	"""	
	def default_initial_input_dict(self):
		return { 'inputs': np.zeros([self.timeslice_size]) }

	"""
	Takes a training batch dict and returns an RNN input dict (by copying out the
	   relevant fields)
	"""
	def batch_to_input_dict(self, batch):
		return { name: batch[name] for name in self.input_shapes.keys() }

	"""
	Takes a training batch dict and returns an distribution conditioning dict (by
	   copying out the relevant fields)
	"""
	def batch_to_condition_dict(self, batch):
		return { name: batch[name] for name in self.condition_shapes.keys() }

	"""
	Takes an input dict and turns it into a single input vector for the RNN
	By default, this just uses the 'inputs' field
	"""
	def input_dict_to_rnn_inputs(self, input_dict):
		return input_dict['inputs']

	"""
	Takes a sampled output time slice, as well as the history of previous input dicts,
	   and returns the next input dict
	By default, just puts the output time slice as the 'inputs' field
	"""
	def sample_to_next_input_dict(self, sample, input_dict_history):
		return { 'inputs': sample }

	"""
	Run the RNN cell over the provided inputs, starting with initial_state
	Returns RNN final state and ouput tensors
	"""
	def run_rnn(self, initial_state, input_dict):
		rnn_inputs = self.input_dict_to_rnn_inputs(input_dict)
		cell = self.rnn_cell()
		outputs, final_state = tf.nn.dynamic_rnn(
			cell, rnn_inputs, initial_state=initial_state, parallel_iterations=1,
			swap_memory=True)
		return final_state, outputs

	@abc.abstractmethod
	def get_step_dist(self, rnn_outputs, condition_dict):
		"""
		Given the output(s) from the RNN, compute the distribution over time slice(s)
		Arguments:
		   - rnn_outputs: a 3D tensor (shape is [batch, time, depth])
		   - condition_dict: a dictionary of tensors that provide extra conditioning info for the
		        distribution.
		Return value:
		   - A Distribution object. Collapses [batch, time] into one dimension and models entries as IID.
		When used for training, batch will be e.g. 128 and time will be the maximum sequence length in the batch.
		When used for sampling, batch will typically be 1 (or more, for e.g. SMC), and time will be 1.
		"""

	"""
	Override of method from Model class
	Assumes that batch contains a 'lengths' and a 'outputs' field
	"""
	def training_loss(self, batch):
		targets = batch['outputs']
		lengths = batch['lengths']

		batch_size = tf.shape(targets)[0]

		_, rnn_outputs = self.run_rnn(self.initial_state(batch_size), self.batch_to_input_dict(batch))
		dist = self.get_step_dist(rnn_outputs, self.batch_to_condition_dict(batch))

		targets_flat = tf.reshape(targets, [-1, self.timeslice_size])

		# Mask out the stuff that was past the end of each training sequence (due to padding)
		mask_flat = tf.reshape(tf.sequence_mask(lengths, dtype=tf.float32), [-1])

		# Compute log probability (We assume that this gives a vector of probabilities, one for each
		#    timeslice entry)
		log_prob = dist.log_prob(targets_flat)

		# Sum across timeslice entries, then across time+batch
		log_prob = tf.reduce_sum(log_prob, 1)
		num_time_slices = tf.to_float(tf.reduce_sum(lengths))
		log_prob = tf.reduce_sum(mask_flat * log_prob) / num_time_slices

		return -log_prob


