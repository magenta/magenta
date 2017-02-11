import abc
from common.datasets.dataset import Dataset
import tensorflow as tf

"""
Base class for sequence datasets
"""
class SequenceDataset(Dataset):

	__metaclass__ = abc.ABCMeta

	def __init__(self, filenames):
		self.filenames = filenames

	@abc.abstractmethod
	def features(self):
		"""
		Should return a feature descriptor dictionary for use with
			tf.parse_single_sequence_example.
		"""

	def load_single(self):
		filenameQueue = tf.train.string_input_producer(self.filenames)
		reader = tf.TFRecordReader()
		_, serializedExample = reader.read(filenameQueue)
		# For now, I'm assuming that we don't use any context features
		_, sequenceFeatures = tf.parse_single_sequence_example(
			serialized = serializedExample,
			sequence_features = self.features()
		)
		anyfeature = sequenceFeatures.values()[0]
		lengths = tf.shape(anyfeature)[0]
		sequenceFeatures['lengths'] = lengths
		return sequenceFeatures

	def load_batch(self, batch_size, num_threads=2):
		sequenceFeatures = self.load_single()
		return tf.train.batch(
			tensors=sequenceFeatures,
			batch_size=batch_size,
			num_threads=num_threads,
			capacity=3*batch_size,
			dynamic_pad=True
		)

