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