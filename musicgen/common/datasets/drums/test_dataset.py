import tensorflow as tf
import common.encoding
from common.datasets import SequenceDataset

path = '/mnt/nfs_datasets/lakh_midi_full/drums_sequence_examples/training_drum_tracks.tfrecord'

input_size = common.encoding.DrumTimeSliceEncoder().output_size
encoder = common.encoding.OneToOneSequenceEncoder(
	common.encoding.IdentityTimeSliceEncoder(input_size)
)

dataset = SequenceDataset([path], encoder)

features = dataset.load_single()

# Run this graph
_features = tf.contrib.learn.run_n(features, n=1)
print(_features[0])