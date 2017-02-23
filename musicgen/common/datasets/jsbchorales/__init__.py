import os
import os.path
from common.datasets.jsbchorales.convert import convert, vec_entry_to_pitch
from common.datasets.sequenceDataset import SequenceDataset
from common.encoding import OneToOneSequenceEncoder, IdentityTimeSliceEncoder

dir_path = os.path.dirname(os.path.realpath(__file__))

def get_dataset(name):
	assert(name in set(['train', 'valid', 'test']))
	filename = dir_path + '/jsb_chorales_' + name + '.tfrecord'
	if not os.path.isfile(filename):
		convert()
	return SequenceDataset([filename], OneToOneSequenceEncoder(
		IdentityTimeSliceEncoder(len(vec_entry_to_pitch()))
	))

def train():
	return get_dataset('train')

def valid():
	return get_dataset('valid')

def test():
	return get_dataset('test')