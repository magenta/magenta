import abc

"""
Abstract base class for all datasets
"""
class Dataset(object):

	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def load_single(self):
		"""
		Load a single entry from the dataset.
		Should return a dictionary of tf Tensors
		"""

	@abc.abstractmethod
	def load_batch(self, batch_size):
		"""
		Load a batch of data from the dataset
		"""