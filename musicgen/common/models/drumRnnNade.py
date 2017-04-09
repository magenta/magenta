import tensorflow as tf
from common.models.rnnNade import RNNNade

"""
An RNN that models a sequence of binary random vectors, where the components of each
   vector are modeled by a NADE whose parameters depend on the RNN hidden state.
Extended specifically for sampling drums with specified conditions.
"""
class DrumRNNNade(RNNNade):

	def __init__(self, hparams, sequence_encoder):
		super(DrumRNNNade, self).__init__(hparams, sequence_encoder)

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
