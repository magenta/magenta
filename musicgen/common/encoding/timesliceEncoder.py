
import abc

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


"""
Encoder that does nothing; just returns whatever representation of the
   timeslice was passed to it
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
