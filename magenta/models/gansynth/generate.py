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
import scipy.io.wavfile as wavfile
import tensorflow as tf

from magenta.models.gansynth.lib import model as lib_model


absl.flags.DEFINE_string('ckpt_dir',
                         '/path/to/ckpt/dir',
                         'Path to the base directory of pretrained checkpoints.'
                         'The base directory should contain many '
                         '"stage_000*" subdirectories.')
absl.flags.DEFINE_string('output_dir',
                         '/path/to/output/dir',
                         'Path to directory to save wave files.')
FLAGS = absl.flags.FLAGS
tfgan = tf.contrib.gan
tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
  absl.flags.FLAGS.alsologtostderr = True

  # Load the model
  model = lib_model.Model.load_from_path(FLAGS.ckpt_dir)

  # Make an output directory if it doesn't exist
  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  # Generate the sounds
  waves = model.generate_samples(12)

  # Write the wave files
  for i in range(len(waves)):
    fname = os.path.join(FLAGS.output_dir, 'generated_{}.wav'.format(i))
    wavfile.write(fname, 16000, waves[i].astype('float32'))



if __name__ == '__main__':
  tf.app.run()
