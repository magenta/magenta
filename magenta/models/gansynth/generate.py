# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate samples with a pretrained GANSynth model.

To use a config of hyperparameters and manual hparams:
>>> python magenta/models/gansynth/generate.py \
>>> --ckpt_dir=/path/to/ckpt/dir --output_dir=/path/to/output/dir
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import absl.flags
import tensorflow as tf

from magenta.models.gansynth.lib import model as lib_model
from magenta.models.gansynth.lib import generate_util as gu
from magenta.models.gansynth.lib import util

absl.flags.DEFINE_string('ckpt_dir',
                         '/tmp/gansynth/acoustic_only',
                         'Path to the base directory of pretrained checkpoints.'
                         'The base directory should contain many '
                         '"stage_000*" subdirectories.')
absl.flags.DEFINE_string('output_dir',
                         '/tmp/gansynth/samples',
                         'Path to directory to save wave files.')
absl.flags.DEFINE_string('midi_file',
                         '',
                         'Path to a MIDI file (.mid) to synthesize.')
absl.flags.DEFINE_integer('batch_size', 8, 'Batch size for generation.')

FLAGS = absl.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
  absl.flags.FLAGS.alsologtostderr = True

  # Load the model
  model = lib_model.Model.load_from_path(FLAGS.ckpt_dir)

  # Make an output directory if it doesn't exist
  output_dir = util.expand_path(FLAGS.output_dir)
  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  if FLAGS.midi_file:
    # If a MIDI file is provided, synthesize interpolations across the clip
    _, notes = gu.load_midi(FLAGS.midi_file)
    # Distribute latent vectors linearly in time
    z_instruments, t_instruments = gu.get_random_instruments(
        model, notes['end_times'][-1])
    # Get latent vectors for each note
    z_notes = gu.get_z_notes(notes, z_instruments, t_instruments)
    # Generate audio for each note
    audio_notes = model.generate_samples_from_z(z_notes, notes['pitches'])
    # Make a single audio clip
    audio_clip = gu.combine_notes(audio_notes,
                                  notes['start_times'],
                                  notes['end_times'])
    waves = [audio_clip]
  else:
    # Otherwise, just generate a batch of random sounds
    waves = model.generate_samples(FLAGS.batch_size)

  # Write the wave files
  for i in range(len(waves)):
    fname = os.path.join(output_dir, 'generated_{}.wav'.format(i))
    gu.save_wav(waves[i], fname)



if __name__ == '__main__':
  tf.app.run()
