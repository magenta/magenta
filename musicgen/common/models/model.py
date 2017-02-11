import abc

"""
Abstract base class for models
"""
class Model(object):

	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def training_loss(self, batch):
		"""
		Takes a batch (a dictionary of tf Tensors) and returns the training loss for that batch
		"""