import abc
import pickle
from magenta.common import HParams

"""
Abstract base class for models
"""
class Model(object):

	__metaclass__ = abc.ABCMeta

	def __init__(self, hparams):
		self.hparams = hparams

	"""
	Save the parameters used to create this model to a pickle file
	"""
	def save(self, filename):
		f = open(filename, 'wb')
		pickle.dump(self.hparams.values(), f)
		f.close()

	"""
	Create a model using the parameters stored in pickled file
	"""
	@classmethod
	def from_file(cls, filename):
		f = open(filename, 'rb')
		keyvals = pickle.load(f)
		f.close()
		hparams = HParams()
		hparams.update(keyvals)
		return cls(hparams)

	@abc.abstractmethod
	def training_loss(self, batch):
		"""
		Takes a batch (a dictionary of tf Tensors) and returns the training loss for that batch
		"""