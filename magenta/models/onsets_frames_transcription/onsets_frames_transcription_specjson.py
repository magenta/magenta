# Copyright 2018 Google Inc. All Rights Reserved.
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

"""Write spectrograms of wav files to JSON.

Usage: onsets_frames_transcription_specjson file1.wav [file2.wav file3.wav]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import tensorflow as tf

from magenta.common import tf_utils
from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription import data
from magenta.models.onsets_frames_transcription import model


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'hparams',
    'onset_mode=length_ms,onset_length=32',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')


def create_spec(filename, hparams):
  """Processes an audio file into a spectrogram."""
  wav_data = tf.gfile.Open(filename).read()
  spec = data.wav_to_spec(wav_data, hparams)
  return spec


def main(argv):
  tf.logging.set_verbosity(FLAGS.log)

  hparams = tf_utils.merge_hparams(
      constants.DEFAULT_HPARAMS, model.get_default_hparams())
  hparams.parse(FLAGS.hparams)

  for filename in argv[1:]:
    tf.logging.info('Generating spectrogram for %s...', filename)

    spec = create_spec(filename, hparams)
    spec_filename = filename + '.json'
    with tf.gfile.Open(spec_filename, 'w') as f:
      f.write(json.dumps(spec.tolist()))
      tf.logging.info('Wrote spectrogram json to %s.', spec_filename)


def console_entry_point():
  tf.app.run(main)

if __name__ == '__main__':
  console_entry_point()
