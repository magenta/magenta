import sys
import os
from common.models import RNNIndependent
from common.sampling import ForwardSample
from common import utils, encoding
from common.datasets.jsbchorales import vec_entry_to_pitch
from magenta.music import sequence_proto_to_midi_file

args = sys.argv[1:]
if len(args) != 1:
	print "Usage: train.py experiment_name"
	sys.exit(1)
experiment_name = args[0]

dir_path = os.path.dirname(os.path.realpath(__file__))
log_dir = dir_path + '/trainOutput/' + experiment_name
utils.ensuredir(log_dir)


model = RNNIndependent.from_file(log_dir + '/model.pickle')
model.hparams.dropout_keep_prob = 1.0

sampler = ForwardSample(model, log_dir, batch_size=10)

# Draw samples that are 40 steps long
samples = sampler.sample(40)

# Convert samples: binaryvec -> pitches -> PolyphonicSequence -> NoteSequence -> MIDI
gen_dir = dir_path + '/generated/' + experiment_name
utils.ensuredir(gen_dir)
steps_per_quarter = 1  # I have no idea what a reasonable value for this is...
for i in range(len(samples)):
	sample = samples[i]
	pitches = encoding.utils.binary_vectors_to_pitches(sample, vec_entry_to_pitch())
	polyseq = encoding.utils.pitches_to_PolyphonicSequence(pitches, steps_per_quarter)
	noteseq = polyseq.to_sequence()
	filename = '{}/sample_{}.mid'.format(gen_dir, i)
	sequence_proto_to_midi_file(noteseq, filename)

print 'Done'