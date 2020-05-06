# Copyright 2020 The Magenta Authors.
# Modifications Copyright 2020 Jack Spencer Smith.
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

import copy
import glob
import json

import absl.flags
import librosa
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from absl import app, logging
from dotmap import DotMap

from magenta.models.polyamp import configs, model_util
from magenta.models.polyamp.dataset_reader import samples_to_cqt
from magenta.models.polyamp.model_util import ModelType, ModelWrapper
from magenta.music import midi_io

absl.flags.DEFINE_string('model_id', None, 'Id to save the model as')

absl.flags.DEFINE_string('config', 'onsets_frames',
                         'Name of the config to use.')
absl.flags.DEFINE_string('model_dir', None,
                         'Path to look for acoustic checkpoints.')
absl.flags.DEFINE_string(
    'hparams', '{}',
    'Json of `name: value` hyperparameter values. '
    'ex. --hparams={\"frames_true_weighing\":2,\"onsets_true_weighing\":8}')
absl.flags.DEFINE_string(
    'transcribed_file_suffix', 'predicted',
    'Optional suffix to add to transcribed files.')
absl.flags.DEFINE_integer(
    'qpm', None,
    'Number of quarters (or beats) per minute. Default: 120')

absl.flags.DEFINE_enum('model_type',
                       ModelType.MELODIC.name,
                       [ModelType.MELODIC.name, ModelType.FULL.name],
                       'type of model to transcribe')
absl.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')
absl.flags.DEFINE_boolean(
    'load_full', False,
    'Whether to use use the weights saved from full training')
FLAGS = absl.flags.FLAGS


def run(argv, config_map):
    """Create transcriptions."""
    logging.set_verbosity(FLAGS.log)

    config = config_map[model_util.ModelType[FLAGS.model_type].value]
    hparams = config.hparams
    hparams.update(json.loads(FLAGS.hparams))
    hparams = DotMap(hparams)
    hparams.model_id = FLAGS.model_id
    hparams.batch_size = 1
    hparams.truncated_length_secs = 0

    model_type = model_util.ModelType[FLAGS.model_type]
    model_wrapper = ModelWrapper(FLAGS.model_dir, type_=model_type, id_=hparams.model_id,
                                 hparams=hparams)

    if model_type is model_util.ModelType.FULL:
        midi_model_wrapper = ModelWrapper(FLAGS.model_dir, ModelType.MELODIC, hparams=hparams)
        midi_model_wrapper.build_model(compile=False)
        midi_model_wrapper.load_newest()
        timbre_model_wrapper = ModelWrapper(FLAGS.model_dir, ModelType.TIMBRE, hparams=hparams)
        timbre_model_wrapper.build_model(compile=False)
        timbre_model_wrapper.load_newest()

        model_wrapper.build_model(midi_model=midi_model_wrapper.get_model(),
                                  timbre_model=timbre_model_wrapper.get_model(),
                                  compile=False)
        model_wrapper.load_newest()
        midi_model_wrapper.load_newest()
    else:
        model_wrapper.build_model(compile=False)
        if FLAGS.load_full:
            full_model_wrapper = ModelWrapper(FLAGS.model_dir, ModelType.FULL, hparams=hparams)
            full_model_wrapper.build_model(
                compile=False,
                midi_model=(model_wrapper.get_model()
                            if model_type is model_util.ModelType.MELODIC
                            else None),
                timbre_model=(model_wrapper.get_model()
                              if model_type is model_util.ModelType.TIMBRE
                              else None)
            )
            full_model_wrapper.load_newest()
        else:
            model_wrapper.load_newest()

    try:
        argv[1].index('*')
        files = glob.glob(argv[1])
    except ValueError:
        files = argv[1:]

    for filename in files:
        logging.info('Starting transcription for %s...', filename)

        samples, sr = librosa.load(filename, hparams.sample_rate)

        if model_type is model_util.ModelType.MELODIC:
            spec = samples_to_cqt(samples, hparams=hparams)
            if hparams.spec_log_amplitude:
                spec = librosa.power_to_db(spec)

            # Add "batch" and channel dims.
            spec = tf.reshape(spec, (1, *spec.shape, 1))

            logging.info('Running inference...')
            sequence_prediction = model_wrapper.predict_from_spec(spec, qpm=FLAGS.qpm)
        else:
            midi_spec = samples_to_cqt(samples, hparams=hparams)
            if hparams.spec_log_amplitude:
                midi_spec = librosa.power_to_db(midi_spec)

            temp_hparams = copy.deepcopy(hparams)
            temp_hparams.spec_hop_length = hparams.timbre_hop_length
            temp_hparams.spec_type = hparams.timbre_spec_type
            temp_hparams.spec_log_amplitude = hparams.timbre_spec_log_amplitude
            timbre_spec = samples_to_cqt(samples, hparams=temp_hparams)
            if hparams.timbre_spec_log_amplitude:
                timbre_spec = librosa.power_to_db(timbre_spec)
                timbre_spec = timbre_spec - librosa.power_to_db(np.array([1e-9]))[0]
                timbre_spec /= K.max(timbre_spec)

            # Add "batch" and channel dims.
            midi_spec = tf.reshape(midi_spec, (1, *midi_spec.shape, 1))
            timbre_spec = tf.reshape(timbre_spec, (1, *timbre_spec.shape, 1))

            logging.info('Running inference...')

            if hparams.present_instruments:
                present_instruments = K.expand_dims(hparams.present_instruments, 0)
            else:
                present_instruments = None

            sequence_prediction = (
                model_wrapper.predict_multi_sequence(midi_spec=midi_spec,
                                                     timbre_spec=timbre_spec,
                                                     present_instruments=present_instruments,
                                                     qpm=FLAGS.qpm)
            )
        midi_filename = filename + FLAGS.transcribed_file_suffix + '.midi'
        midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)

        logging.info('Transcription written to %s.', midi_filename)


def main(argv):
    run(argv, config_map=configs.CONFIG_MAP)


def console_entry_point():
    app.run(main)


if __name__ == '__main__':
    console_entry_point()
