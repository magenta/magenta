import numpy as np
from magenta.models.polyphony_rnn.polyphony_lib import PolyphonicEvent, PolyphonicSequence

"""
Converts a pitch tuple into a binary vector representation
'vec_entry_to_pitch' is a list of the same length as the binary vector
   to be returned. It says which pitch value gets mapped to which vector element.
   (This allows for making binary vectors out of very sparse pitch sets, e.g. drums)
"""
def _pitches_to_binary_vector(pitches, vec_entry_to_pitch):
	n = len(vec_entry_to_pitch)
	pitch_to_vec_entry = {vec_entry_to_pitch[i]: i for i in range(n)}
	binvec = [0] * n
	for pitch in pitches:
		binvec[pitch_to_vec_entry[pitch]] = 1
	return np.array(binvec)


"""
Converts a list of pitch tuples into a list of binary vectors
"""
def pitches_to_binary_vectors(pitches_list, vec_entry_to_pitch):
	return [_pitches_to_binary_vector(pitches, vec_entry_to_pitch) for pitches in pitches_list]


"""
Converts a binary vector to a pitch tuple
"""
def _binary_vector_to_pitches(vec, vec_entry_to_pitch):
	lst = []
	for i in range(len(vec)):
		if vec[i] == 1:
			lst.append(vec_entry_to_pitch[i])
	return tuple(lst)


"""
Converts a list of binary vectors to a list of pitch tuples
"""
def binary_vectors_to_pitches(vec_list, vec_entry_to_pitch):
	return [_binary_vector_to_pitches(vec, vec_entry_to_pitch) for vec in vec_list]


"""
Converts a list of pitch tuples into a magenta PolyphonicSequence.
Assumes that contiguous notes are sustained.
"""
def pitches_to_PolyphonicSequence(pitches_list, steps_per_quarter, start_step=0):
    events = PolyphonicSequence(steps_per_quarter=steps_per_quarter, start_step=start_step)
    events.append(PolyphonicEvent(event_type=PolyphonicEvent.START, pitch=None))

    active_pitches = []
    for pitches in pitches_list:
    	step_events = []

    	# Remove pitches that are no longer active (b/c we no longer see them in the current pitches)
    	for pitch in active_pitches:
    		if pitch not in pitches:
    			active_pitches.remove(pitch)

    	for pitch in pitches:
    		if pitch in active_pitches:
    			step_events.append(PolyphonicEvent(event_type=PolyphonicEvent.CONTINUED_NOTE, pitch=pitch))
    		else:
    			active_pitches.append(pitch)
    			step_events.append(PolyphonicEvent(event_type=PolyphonicEvent.NEW_NOTE, pitch=pitch))

    	step_events = sorted(step_events, key=lambda e: e.pitch, reverse=True)
    	for event in step_events:
    		events.append(event)

      	events.append(PolyphonicEvent(event_type=PolyphonicEvent.STEP_END, pitch=None))

    events.append(PolyphonicEvent(event_type=PolyphonicEvent.END, pitch=None))

    return events