import os
import sys
import pickle
import tensorflow as tf
from common.encoding import utils, OneToOneSequenceEncoder, IdentityTimeSliceEncoder

dir_path = os.path.dirname(os.path.realpath(__file__))

def load_pickle():
	filename = dir_path + '/JSB Chorales.pickle'
	f = open(filename, 'r')
	data = pickle.load(f)
	f.close()
	return data

_vec_entry_to_pitch = None
def vec_entry_to_pitch():
	global _vec_entry_to_pitch
	if _vec_entry_to_pitch is None:
		data = load_pickle()
		# Compute the min and max pitch values
		minpitch = sys.maxint
		maxpitch = 0
		for _,songs in data.iteritems():
			for song in songs:
				for pitchTuple in song:
					if len(pitchTuple) > 0:
						maxpitch = max(maxpitch, max(pitchTuple))
						minpitch = min(minpitch, min(pitchTuple))
		vecsize = maxpitch - minpitch + 1
		_vec_entry_to_pitch = [0] * vecsize
		curr_pitch = minpitch
		for i in range(vecsize):
			_vec_entry_to_pitch[i] = curr_pitch
			curr_pitch += 1
	return _vec_entry_to_pitch

def convert(addEmptyStartTuple=True):
	data = load_pickle()

	if addEmptyStartTuple:
		# Add an empty tuple to start every song (so we can always use "all notes off" as
		#    an RNN initial state)
		for _,songs in data.iteritems():
			for song in songs:
				song.insert(0, ())

	# Convert all train, test, validation sets into binary vector representation
	binvecData = {
		setname: map(lambda song: utils.pitches_to_binary_vectors(song, vec_entry_to_pitch()), songs) for setname, songs in data.iteritems()
	}

	# Write out a separate TFRecord file for each of train, test, validation sets
	sequence_encoder = OneToOneSequenceEncoder(
		IdentityTimeSliceEncoder(len(vec_entry_to_pitch()))
	)
	for setname,songs in binvecData.iteritems():
		print "Converting {} set...".format(setname)
		filename = dir_path + "/jsb_chorales_{}.tfrecord".format(setname)
		writer = tf.python_io.TFRecordWriter(filename)
		for song in songs:
			ex = sequence_encoder.encode(song)
			writer.write(ex.SerializeToString())
		writer.close()

def test():
	filename = dir_path + '/jsb_chorales_test.tfrecord'

	sequence_encoder = OneToOneSequenceEncoder(
		IdentityTimeSliceEncoder(len(vec_entry_to_pitch()))
	)

	# Build graph that'll load up and parse an example from the .tfrecord
	filenameQueue = tf.train.string_input_producer([filename])
	reader = tf.TFRecordReader()
	_, serializedExample = reader.read(filenameQueue)
	sequenceFeatures = sequence_encoder.parse(serializedExample)

	# Run this graph
	seqFeatures = tf.contrib.learn.run_n(sequenceFeatures, n=1)
	print(seqFeatures[0])


if __name__ == '__main__':
	convert()
	# test()
