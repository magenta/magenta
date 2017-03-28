import tensorflow as tf
from common.models.rnnIndependent import RNNIndependent

"""
An RNN that models a sequence of binary random vectors, where the components of each
   vector are modeled independently from one other given the RNN hidden state.

Extended specifically for sampling drums with specified conditions.
"""
class DrumRNNIndependent(RNNIndependent):

	def __init__(self, hparams, sequence_encoder):
		super(DrumRNNIndependent, self).__init__(hparams, sequence_encoder)

	@property
	def condition_shapes(self):
		return {'known_notes': [9]}

	# Returns 0 or negative infinity if sample satisfies or doesn't satisfy condition.
	def eval_factor_function(self, sample, condition):
		if len(condition) == 0:
			return 0

		for index in range(len(condition)):
			if index == len(condition) - 1:
				if condition[index] == -1 or condition[index] == sample[index]:
					return 0
			elif condition[index] == -1:
				continue
			elif condition[index] != sample[index]:
				return float('-inf')
		
		return 0