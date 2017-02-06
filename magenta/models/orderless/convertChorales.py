import sys
import pickle
import tensorflow as tf

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

def main():
	filename = 'data/JSB Chorales.pickle'
	f = open(filename, 'r')

	data = pickle.load(f)

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

	# Converts a pitch tuple into a binary vector representation
	def pitchTupleToBinVec(pitchTuple):
		binvec = [0] * vecsize
		for pitch in pitchTuple:
			binvec[pitch - minpitch] = 1
		return binvec

	# Convert all train, test, validation sets into binary vector representation
	binvecData = {
		setname: map(lambda song: map(pitchTupleToBinVec, song), songs) for setname, songs in data.iteritems()
	}

	# Write out a separate TFRecord file for each of train, test, validation sets
	for setname,songs in binvecData.iteritems():
		print "Converting {} set...".format(setname)
		filename = "data/jsb_chorales_{}.tfrecord".format(setname)
		writer = tf.python_io.TFRecordWriter(filename)
		for song in songs:
			ex = makeSequenceExample(song)
			writer.write(ex.SerializeToString())
		writer.close()

def test():
	filename = 'data/jsb_chorales_test.tfrecord'

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
	main()
	# test()
