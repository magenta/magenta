import sys
import os
from common.models import RNNIndependent
from common.datasets import jsbchorales
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
	timeslice_size = len(vec_entry_to_pitch()),

	# rnn_layer_sizes = [128],
	# dropout_keep_prob = 1.0,
	# attn_length =  None,

	rnn_layer_sizes = [128, 128],
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

model = RNNIndependent(model_params)
model.save(log_dir + '/model.pickle')

dataset = jsbchorales.train()

training.train(model, dataset, train_params)