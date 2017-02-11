import os
from common.models import RNNIndependent
from common.datasets import jsbchorales
from common import training
from magenta.common import HParams

dir_path = os.path.dirname(os.path.realpath(__file__))

hparams = HParams(
	# Data prep
	num_threads = 2,
	batch_size = 128,

	# Model
	timeslice_size = 54,
	rnn_layer_sizes = [128],
	dropout_keep_prob = 1.0,
	attn_length =  None,

	# Training
	n_training_iters = 20000,
	initial_learning_rate = 0.001,
	decay_steps = 1000,
	decay_rate = 0.95,
	gradient_clip_norm = 5,
	summary_frequency = 10,
	log_dir = dir_path + '/trainOutput',
	save_model_secs = 30
)

model = RNNIndependent(hparams)
dataset = jsbchorales.train()

training.train(model, dataset, hparams)