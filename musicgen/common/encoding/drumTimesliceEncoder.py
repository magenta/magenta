from common.encoding.timesliceEncoder import TimeSliceEncoder
from magenta.music.drums_encoder_decoder import DEFAULT_DRUM_TYPE_PITCHES as drum_type_pitches
import numpy as np

"""
Adaptation of magenta/music/drums_encoder_decoder.py/MultiDrumOneHotEncoding
Converts sets of pitches into multi-hot binary vectors
"""
class DrumTimeSliceEncoder(TimeSliceEncoder):

	def __init__(self):
		self._drum_map = dict(enumerate(drum_type_pitches))
		self._inverse_drum_map = dict((pitch, index)
								  for index, pitches in self._drum_map.items()
								  for pitch in pitches)

	@property
	def output_size(self):
		return len(drum_type_pitches)

	def encode(self, pitches):
		vec = np.zeros([self.output_size])
		for pitch in pitches:
			if pitch in self._inverse_drum_map:
				index = self._inverse_drum_map[pitch]
				vec[index] = 1
		return vec

	def decode(self, binvector):
		pitches = []
		for i in range(len(binvector)):
			if binvector[i] == 1:
				possible_pitches_for_i = self._drum_map[i]
				# Use the first possible pitch
				pitches.append(possible_pitches_for_i[0])
		return tuple(pitches)

