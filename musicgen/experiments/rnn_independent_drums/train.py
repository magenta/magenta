import sys
import os
from common.models import RNNIndependent
from common.datasets import SequenceDataset
import common.encoding as encoding
from common import training
from common import utils
from common.datasets.jsbchorales import vec_entry_to_pitch
from magenta.common import HParams

args = sys.argv[1:]
if len(args) != 1:
	print "Usage: train.py experiment_name"
	sys.exit(1)
experiment_name = args[0]

dir_path = os.path.dirname(os.path.realpath(__file__))
log_dir = dir_path + '/trainOutput/' + experiment_name
utils.ensuredir(log_dir)

model_params = HParams(
	rnn_layer_sizes = [256, 256, 256],
	dropout_keep_prob = 0.5,
	attn_length =  None,
)

train_params = HParams(
	num_threads = 2,
	batch_size = 128,

	n_training_iters = 20000,
	initial_learning_rate = 0.001,
	decay_steps = 1000,
	decay_rate = 0.95,
	gradient_clip_norm = 5,
	summary_frequency = 10,
	log_dir = log_dir,
	save_model_secs = 30
)

timeslice_encoder = encoding.IdentityTimeSliceEncoder(encoding.DrumTimeSliceEncoder().output_size)

# data_filename = '/mnt/nfs_datasets/lakh_midi_full/drums_prependEmpty/training_drum_tracks.tfrecord'
# sequence_encoder = encoding.OneToOneSequenceEncoder(timeslice_encoder)

data_filename = '/mnt/nfs_datasets/lakh_midi_full/drums_lookback_meter/training_drum_tracks.tfrecord'
sequence_encoder = encoding.LookbackSequenceEncoder(timeslice_encoder,
	lookback_distances=[],
	binary_counter_bits=6
)

dataset = SequenceDataset([data_filename], sequence_encoder)

features = dataset.load_single()

model = RNNIndependent(model_params, sequence_encoder)
model.save(log_dir + '/model.pickle')

training.train(model, dataset, train_params)

