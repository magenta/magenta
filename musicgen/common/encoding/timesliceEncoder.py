
import abc
import numpy as np

"""
Generalization of magenta's OneHotEncoding, so we can move beyond one-hot
   'event' prediction to more general representations of time slices.
"""
class TimeSliceEncoder(object):

	__metaclass__ = abc.ABCMeta

	@abc.abstractproperty
	def output_size(self):
		pass

	@abc.abstractmethod
  	def encode(self, timeslice):
  		pass

  	@abc.abstractmethod
  	def decode(self, encoded_timeslice):
  		""" Undoes the effect of encode """
  		pass

  	@property
  	def empty_timeslice(self):
  		""" Encoding of an empty timeslice """
  		return np.zeros([self.output_size])


"""
Encoder that does nothing; just returns whatever representation of the
   timeslice was passed to it
Intended to pass-through binary vectors
"""
class IdentityTimeSliceEncoder(TimeSliceEncoder):

	def __init__(self, size):
		self.size = size

	@property
	def output_size(self):
		return self.size

	def encode(self, timeslice):
		return timeslice

	def decode(self, encoded_timeslice):
		return encoded_timeslice
