
import abc
import tensorflow as tf
from magenta.music.encoder_decoder import DEFAULT_LOOKBACK_DISTANCES
import numpy as np

"""
Like magenta's EventSequenceEncoderDecoder, but for our codebase
"""
class SequenceEncoder(object):

	__metaclass__ = abc.ABCMeta

	def __init__(self, timeslice_encoder):
		self.timeslice_encoder = timeslice_encoder

	@property
	def encoded_timeslice_size(self):
		return self.timeslice_encoder.output_size

	@abc.abstractproperty
	def rnn_input_size(self):
		pass

	@abc.abstractproperty
	def condition_shapes(self):
		"""
		Returns a dictionary from names of extra condition info to their shapes.
		(This should be empty if there's no conditioning)
		"""
		pass

	@abc.abstractmethod
	def rnn_input_for_timeslice(self, encoded_timeslices, index, condition_dict=None):
		pass

	def encode(self, timeslices, condition_dicts=None):
		"""
		Turn the list of timeslices (and condition_dicts) into a tensorflow SequenceExample
		"""
		if condition_dicts is not None:
			assert(len(timeslices) == len(condition_dicts)+1)
		n = len(timeslices)

		# Encoded time slices
		encoded_timeslices = [self.timeslice_encoder.encode(timeslice) for timeslice in timeslices]

		# Get features for sequence example
		features = {
			'inputs': [self.rnn_input_for_timeslice(encoded_timeslices, i, (None if (condition_dicts is None) else condition_dicts[i])) for i in range(0,n-1)],
			'outputs': [encoded_timeslices[i] for i in range(1,n)]
		}
		if condition_dicts is not None:
			for name,_ in condition_dicts[0].iteritems():
				features[name] = [cdict[name] for cdict in condition_dicts]

		# Convert features to SequenceExample protobuf format
		featureList = {
			name: tf.train.FeatureList(feature=[
				tf.train.Feature(float_list=tf.train.FloatList(value=f)) for f in flist
			]) for name,flist in features.iteritems()
		}
		featureLists = tf.train.FeatureLists(feature_list=featureList)
		return tf.train.SequenceExample(feature_lists=featureLists)

	def parse(self, serializedExample):
		"""
		Parse a serialized SequenceExample (as part of a TF graph)
		"""
		# Put together a map from feature name to shape
		features = {
			'inputs': [self.rnn_input_size],
			'outputs': [self.encoded_timeslice_size]
		}
		for name,shape in self.condition_shapes:
			features[name] = shape

		# Convert these into tf.FixedLenSequenceFeature
		featureDescriptors = {
			name: tf.FixedLenSequenceFeature(shape, dtype=tf.float32) for name,shape in features.iteritems()
		}

			# For now, I'm assuming that we don't use any context features
		_, sequenceFeatures = tf.parse_single_sequence_example(
			serialized = serializedExample,
			sequence_features = featureDescriptors
		)
		return sequenceFeatures


"""
Uses the encoding of one time slice to predict the encoding of the next time slice
"""
class OneToOneSequenceEncoder(SequenceEncoder):

	def __init__(self, timeslice_encoder):
		super(OneToOneSequenceEncoder, self).__init__(timeslice_encoder)

	@property
	def rnn_input_size(self):
		return self.encoded_timeslice_size

	@property
	def condition_shapes(self):
		return {}

	def rnn_input_for_timeslice(self, encoded_timeslices, index, condition_dict=None):
		return encoded_timeslices[index]


"""
Uses the encoding of the current time slice, some number of prevous time slices, and a meter counter
	 to predict the next time slice.
Based on magenta/music/encoder_decoder.py/LookbackEventSequenceEncoderDecoder
"""
class LookbackSequenceEncoder(SequenceEncoder):

	def __init__(self, timeslice_encoder, lookback_distances=None, binary_counter_bits=5):
		super(LookbackSequenceEncoder, self).__init__(timeslice_encoder)
		self._lookback_distances = (lookback_distances
																if lookback_distances is not None
																else DEFAULT_LOOKBACK_DISTANCES)
		self._binary_counter_bits = binary_counter_bits

	@property
	def rnn_input_size(self):
		timeslice_size = self.timeslice_encoder.output_size
		num_lookbacks = len(self._lookback_distances)
		return (timeslice_size +                  # current event
						num_lookbacks * timeslice_size +  # next event for each lookback
						self._binary_counter_bits +       # binary counters
						num_lookbacks)                    # whether event matches lookbacks

	@property
	def condition_shapes(self):
		return {}

	def rnn_input_for_timeslice(self, encoded_timeslices, index, condition_dict=None):

		input_ = np.zeros([self.rnn_input_size])
		offset = 0

		# The current timeslice
		timeslice_size = self.timeslice_encoder.output_size
		input_[offset:offset+timeslice_size] = encoded_timeslices[index]
		offset += timeslice_size

		# Lookback timeslices
		for i, lookback_distance in enumerate(self._lookback_distances):
			lookback_index = index - lookback_distance + 1
			if lookback_index < 0:
				input_[offset:offset+timeslice_size] = self.timeslice_encoder.empty_timeslice
			else:
				input_[offset:offset+timeslice_size] = encoded_timeslices[lookback_index]
			offset += timeslice_size

		# Binary time counter given the metric location of the *next* timeslice
		# (Encoded as {-1, 1} instead of {0, 1}...not sure why...)
		n = index + 1
		for i in range(self._binary_counter_bits):
			input_[offset] = 1.0 if (n / 2 ** i) % 2 else -1.0
			offset += 1

		# Boolean flags indicating whether the current timeslice was repeating at
		#    the provided lookback distances
		for i, lookback_distance in enumerate(self._lookback_distances):
			lookback_index = index - lookback_distance + 1
			if (lookback_index >= 0 and
					np.all(encoded_timeslices[index] == encoded_timeslices[lookback_index])):
				input_[offset] = 1.0
			offset += 1

		assert offset == self.rnn_input_size

		return input_




