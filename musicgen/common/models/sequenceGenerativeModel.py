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

	def __init__(self, hparams, sequence_encoder):
		super(SequenceGenerativeModel, self).__init__(hparams)
		self.sequence_encoder = sequence_encoder
		self._rnn_cell = None

	@classmethod
	def from_file(cls, filename, sequence_encoder):
		hparams = Model.hparams_from_file(filename)
		return cls(hparams, sequence_encoder)

	@property
  	def timeslice_size(self):
		return self.sequence_encoder.encoded_timeslice_size

	@property
	def rnn_input_size(self):
		return self.sequence_encoder.rnn_input_size

	"""
	Names and shapes of all the conditioning data this model expects in its condition dicts
	"""
	@property
	def condition_shapes(self):
		return self.sequence_encoder.condition_shapes

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
	Initial timeslice to use for input to this model in the absence of any priming inputs.
	By default, this uses the encoder's empty timeslice (which is a zero vector)
	"""	
	def default_initial_timeslice(self):
		return self.sequence_encoder.timeslice_encoder.empty_timeslice

	"""
	Takes a training batch dict and returns an distribution conditioning dict (by
	   copying out the relevant fields)
	"""
	def batch_to_condition_dict(self, batch):
		return { name: batch[name] for name in self.condition_shapes.keys() }

	"""
	Takes a history of time slices, plus the current conditioning dict, and
	   returns the next input vector to the RNN.
	"""
	def next_rnn_input(self, timeslice_history, condition_dict):
		index = len(timeslice_history) - 1
		return self.sequence_encoder.rnn_input_for_timeslice(timeslice_history, index, condition_dict)

	"""
	Run the RNN cell over the provided input vector, starting with initial_state
	Returns RNN final state and ouput tensors
	"""
	def run_rnn(self, initial_state, rnn_inputs):
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
	Given the sample for the current timeslice and a condition dictionary, return a score in log-space.
	Sampling algorithms, such as particle filtering, can take this into account.
	By default, returns 0. Subclasses can override this behavior.

	Condition dict has only one key and value pair.
	"""
	def eval_factor_function(self, sample, condition_dict):
		if len(condition_dict) == 0:
			return 1
		condition = condition_dict["known_notes"]
		for index in range(len(condition)):
			if condition[index] == -1:
				continue
			elif condition[index] != sample[index]:
				return 0
		return 1

	"""
	Override of method from Model class
	Assumes that batch contains a 'lengths' and a 'outputs' field
	NOTE: During training, we assume that timeslices + conditioning info has already been processed into
	   a single, unified RNN input vector, which is provided as the 'inputs' field of the batch.
	   Conditioning info is still separately available for building timeslice distributions.
	"""
	def training_loss(self, batch):
		inputs = batch['inputs']
		targets = batch['outputs']
		lengths = batch['lengths']

		batch_size = tf.shape(targets)[0]

		_, rnn_outputs = self.run_rnn(self.initial_state(batch_size), inputs)
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


