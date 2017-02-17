import os
import sys
import pickle
import tensorflow as tf
from common import representation as rep

dir_path = os.path.dirname(os.path.realpath(__file__))

# Make a sequence example from a single song in binary vec list format
# The sequence example stores (input, output) at each time step, where
#    output is just the next vector in the sequence
# For the time being, I'm just saving both the input and output as FloatLists
def makeSequenceExample(song):
	inputs = song[:len(song)-1]
	outputs = song[1:]
	inFeatures = [tf.train.Feature(float_list=tf.train.FloatList(value=inp)) for inp in inputs]
	outFeatures = [tf.train.Feature(float_list=tf.train.FloatList(value=out)) for out in outputs]
	featureList = {
		'inputs': tf.train.FeatureList(feature=inFeatures),
		'outputs': tf.train.FeatureList(feature=outFeatures)
	}
	featureLists = tf.train.FeatureLists(feature_list=featureList)
	return tf.train.SequenceExample(feature_lists=featureLists)

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
		setname: map(lambda song: rep.pitches_to_binary_vectors(song, vec_entry_to_pitch()), songs) for setname, songs in data.iteritems()
	}

	# Write out a separate TFRecord file for each of train, test, validation sets
	for setname,songs in binvecData.iteritems():
		print "Converting {} set...".format(setname)
		filename = dir_path + "/jsb_chorales_{}.tfrecord".format(setname)
		writer = tf.python_io.TFRecordWriter(filename)
		for song in songs:
			ex = makeSequenceExample(song)
			writer.write(ex.SerializeToString())
		writer.close()

def test():
	filename = dir_path + '/jsb_chorales_test.tfrecord'

	# Build graph that'll load up and parse an example from the .tfrecord
	filenameQueue = tf.train.string_input_producer([filename])
	reader = tf.TFRecordReader()
	_, serializedExample = reader.read(filenameQueue)
	_, sequenceFeatures = tf.parse_single_sequence_example(
		serialized = serializedExample,
		sequence_features = {
			'inputs': tf.FixedLenSequenceFeature([54], dtype=tf.float32),
			'outputs': tf.FixedLenSequenceFeature([54], dtype=tf.float32)
		}
	)

	# Run this graph
	seqFeatures = tf.contrib.learn.run_n(sequenceFeatures, n=1)
	print(seqFeatures[0])


if __name__ == '__main__':
	convert()
	# test()
