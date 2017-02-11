from common.datasets.sequenceDataset import SequenceDataset
import tensorflow as tf

"""
A sequence dataset designed for tasks where we predict the next tensor in a sequence
   given the previous tensors in the sequence.
Practically, this means that the input and the output at each time slice have the
   same size.
"""
class AutoregressiveSequenceDataset(SequenceDataset):

	def __init__(self, filenames, tensorSize):
		super(AutoregressiveSequenceDataset, self).__init__(filenames)
		self.size = tensorSize

	def features(self):
		return {
			'inputs': tf.FixedLenSequenceFeature(self.size, dtype=tf.float32),
			'outputs': tf.FixedLenSequenceFeature(self.size, dtype=tf.float32)
		}
