# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transcribe a recording of piano audio."""

from __future__ import absolute_import, division, print_function

import copy
import glob
import json

import librosa
import tensorflow.compat.v1 as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap

from magenta.models.polyamp.data import wav_to_spec_op, samples_to_cqt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('using_plaidml', False, 'Are we using plaidml')

tf.app.flags.DEFINE_string('model_id', None, 'Id to save the model as')

tf.app.flags.DEFINE_string('config', 'onsets_frames',
                           'Name of the config to use.')
tf.app.flags.DEFINE_string('model_dir', None,
                           'Path to look for acoustic checkpoints.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'Filename of the checkpoint to use. If not specified, will use the latest '
    'checkpoint')
tf.app.flags.DEFINE_string(
    'hparams',
    '{}',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_boolean(
    'load_audio_with_librosa', False,
    'Whether to use librosa for sampling audio (required for 24-bit audio)')
tf.app.flags.DEFINE_string(
    'transcribed_file_suffix', 'predicted',
    'Optional suffix to add to transcribed files.')
tf.app.flags.DEFINE_integer(
    'qpm', None,
    'Number of quarters (or beats) per minute. Default: 120')

tf.app.flags.DEFINE_enum('model_type', 'MIDI', ['MIDI', 'FULL'],
                         'type of model to transcribe')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')
tf.app.flags.DEFINE_boolean(
    'load_full', False,
    'Whether to use use the weights saved from full training')

from magenta.models.polyamp import configs, model_util
from magenta.models.polyamp import data
from magenta.models.polyamp.model_util import ModelWrapper, ModelType
from magenta.music import midi_io


def run(argv, config_map, data_fn):
    """Create transcriptions."""
    tf.compat.v1.logging.set_verbosity(FLAGS.log)

    config = config_map[FLAGS.config]
    hparams = config.hparams
    # For this script, default to not using cudnn.
    hparams.update(json.loads(FLAGS.hparams))
    hparams = DotMap(hparams)
    hparams.use_cudnn = False
    hparams.model_id = FLAGS.model_id
    hparams.batch_size = 1
    hparams.truncated_length_secs = 0

    model_type = model_util.ModelType[FLAGS.model_type]
    model = ModelWrapper('E:/models', type=model_type, id=hparams.model_id, hparams=hparams)

    if model_type is model_util.ModelType.FULL:

        midi_model = ModelWrapper('E:/models', ModelType.MIDI, hparams=hparams)
        midi_model.build_model(compile=False)
        midi_model.load_newest()
        timbre_model = ModelWrapper('E:/models', ModelType.TIMBRE, hparams=hparams)
        timbre_model.build_model(compile=False)
        timbre_model.load_newest()

        model.build_model(midi_model=midi_model.get_model(),
                          timbre_model=timbre_model.get_model(),
                          compile=False)
        model.load_newest()
        midi_model.load_newest()


    else:
        model.build_model(compile=False)
    try:
        if FLAGS.load_full:
            full = ModelWrapper('E:/models', ModelType.FULL,
                                hparams=hparams)
            full.build_model(compile=False,
                             midi_model=model.get_model()
                             if model_type is model_util.ModelType.MIDI
                             else None,
                             timbre_model=model.get_model()
                             if model_type is model_util.ModelType.TIMBRE
                             else None)
            full.load_newest()
        else:
            model.load_newest()
    except:
        pass

    try:
        argv[1].index('*')
        files = glob.glob(argv[1])
    except ValueError:
        files = argv[1:]

    for filename in files:
        tf.compat.v1.logging.info('Starting transcription for %s...', filename)

        # wav_data = tf.gfile.Open(filename, 'rb').read()
        samples, sr = librosa.load(filename, hparams.sample_rate)

        if model_type is model_util.ModelType.MIDI:
            # spec = wav_to_spec_op(wav_data, hparams=hparams)
            spec = samples_to_cqt(samples, hparams=hparams)
            if hparams.spec_log_amplitude:
                spec = librosa.power_to_db(spec)

            # add "batch" and channel dims
            spec = tf.reshape(spec, (1, *spec.shape, 1))

            tf.compat.v1.logging.info('Running inference...')
            sequence_prediction = model.predict_from_spec(spec, qpm=FLAGS.qpm)
        else:
            # midi_spec = wav_to_spec_op(wav_data, hparams=hparams)
            midi_spec = samples_to_cqt(samples, hparams=hparams)
            if hparams.spec_log_amplitude:
                midi_spec = librosa.power_to_db(midi_spec)

            temp_hparams = copy.deepcopy(hparams)
            temp_hparams.spec_hop_length = hparams.timbre_hop_length
            temp_hparams.spec_type = hparams.timbre_spec_type
            temp_hparams.spec_log_amplitude = hparams.timbre_spec_log_amplitude
            # timbre_spec = wav_to_spec_op(wav_data, hparams=temp_hparams)
            timbre_spec = samples_to_cqt(samples, hparams=temp_hparams)
            if hparams.timbre_spec_log_amplitude:
                timbre_spec = librosa.power_to_db(timbre_spec)
                timbre_spec = timbre_spec - librosa.power_to_db(np.array([1e-9]))[0]
                timbre_spec /= K.max(timbre_spec)

            # add "batch" and channel dims
            midi_spec = tf.reshape(midi_spec, (1, *midi_spec.shape, 1))
            timbre_spec = tf.reshape(timbre_spec, (1, *timbre_spec.shape, 1))

            tf.compat.v1.logging.info('Running inference...')

            if hparams.present_instruments:
                present_instruments = K.expand_dims(hparams.present_instruments, 0)
            else:
                present_instruments = None

            sequence_prediction = model.predict_multi_sequence(midi_spec=midi_spec,
                                                               timbre_spec=timbre_spec,
                                                               present_instruments=present_instruments,
                                                               qpm=FLAGS.qpm)
        # assert len(prediction_list) == 1

        # sequence_prediction = music_pb2.NoteSequence.FromString(sequence_prediction)

        midi_filename = filename + FLAGS.transcribed_file_suffix + '.midi'
        midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)

        tf.compat.v1.logging.info('Transcription written to %s.', midi_filename)


def main(argv):
    run(argv, config_map=configs.CONFIG_MAP, data_fn=data.provide_batch)


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
