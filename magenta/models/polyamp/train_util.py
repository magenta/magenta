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

"""Utilities for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import functools
import glob
import os
import random
import sys

import six
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

# Should not be called from within the graph to avoid redundant summaries.
from magenta.models.polyamp import audio_label_data_utils, constants
from magenta.models.polyamp.callback import EvaluationMetrics, \
    MidiPredictionMetrics
from magenta.models.polyamp.dataset_reader import wav_to_spec_op
from magenta.models.polyamp.data_generator import DataGenerator
from magenta.models.polyamp.metrics import f1_score
from magenta.models.polyamp.model_util import ModelWrapper, ModelType
from magenta.models.polyamp.timbre_dataset_reader import create_spectrogram
from magenta.music import midi_io, audio_io
from magenta.music.protobuf import music_pb2


def create_example(filename, sample_rate, load_audio_with_librosa):
    """Processes an audio file into an Example proto."""
    wav_data = tf.compat.v1.gfile.Open(filename, 'rb').read()
    example_list = list(
        audio_label_data_utils.process_record(
            wav_data=wav_data,
            sample_rate=sample_rate,
            ns=music_pb2.NoteSequence(),
            # decode to handle filenames with extended characters.
            example_id=six.ensure_text(filename, 'utf-8'),
            min_length=0,
            max_length=-1,
            allow_empty_notesequence=True,
            load_audio_with_librosa=load_audio_with_librosa))
    assert len(example_list) == 1
    return example_list[0].SerializeToString()


def _trial_summary(hparams, model_dir, output_dir, additional_trial_info):
    """Writes a tensorboard text summary of the trial."""

    summaries_to_write = collections.OrderedDict()
    summaries_to_write['model_dir'] = model_dir
    summaries_to_write['command_line_args'] = ' \\'.join(sys.argv)

    tf.compat.v1.logging.info('Writing hparams summary: %s', hparams)

    # Create a markdown table from hparams.
    header = '| Key | Value |\n| :--- | :--- |\n'
    keys = sorted(hparams.keys())
    lines = ['| %s | %s |' % (key, str(hparams[key])) for key in keys]
    hparams_table = header + '\n'.join(lines) + '\n'

    summaries_to_write['hparams'] = hparams_table

    summaries_to_write.update(additional_trial_info)

    with tf.compat.v1.Session() as sess:
        writer = tf.compat.v1.summary.FileWriter(output_dir, graph=sess.graph)
        for name, summary in summaries_to_write.items():
            tf.compat.v1.logging.info('Writing summary for %s: %s', name, summary)
            writer.add_summary(
                tf.compat.v1.summary.text(name, tf.constant(summary, name=name),
                                          collections=[]).eval())
        writer.close()


def train(data_fn,
          model_dir,
          model_type,
          preprocess_examples,
          hparams,
          num_steps=50):
    """Train loop."""

    transcription_data = functools.partial(
        data_fn,
        preprocess_examples=preprocess_examples,
        is_training=True,
        shuffle_examples=True,
        skip_n_initial_records=random.randint(0, 128))

    model = ModelWrapper(model_dir, model_type, id_=hparams.model_id,
                         dataset=transcription_data(params=hparams),
                         batch_size=hparams.batch_size, steps_per_epoch=hparams.epochs_per_save,
                         hparams=hparams)
    # midi_model.load_model(71.85, 74.98)
    # midi_model.load_model(74.27, 70.17)
    # midi_model.load_model(91.46, 92.58, 'no-weight')
    # midi_model.load_model(69.19, 81.61, 'no-weight')
    # midi_model.load_model(76.70, 82.13, 'no-loops')
    # midi_model.load_model(83.89, 89.36, 'no-weight')
    # midi_model.load_model(78.74, 83.30, 'dataset-test')
    # midi_model.load_model(80.37, 83.94, 'weights-zero')
    # midi_model.load_model(70.07, 80.87, 'weights-zero')
    # midi_model.load_model(71.05, 85.00, 'frame-weight-4') #fp:57, fr: 92, op:87, or: 82
    # midi_model.load_model(70.11, 84.78, '3-4-9-threshold')
    # model.load_model(0.0, id='901dcedede0e40898ba0daf790673b4c')
    # model.load_model(8.93, id='eadb12c34708460896a671a7a3dabf58')
    # model.load_model(78.67, id='temp')
    # model.load_model(37.50, id='training-time')
    # model.load_model(15.62, id="shared")
    # model.load_model(11.10, id='shared-512', epoch_num=11)
    # model.load_model(7.19, id='parallel-first', epoch_num=3)
    # model.load_model(0, id='long', epoch_num=6)
    # model.load_model(20.64, id='2-glob', epoch_num=38)
    # model.load_model(17.81, id='2-glob', epoch_num=27)

    if model_type == ModelType.MELODIC:
        # model.load_model(43.15, 58.60, id='big-lstm', epoch_num=98)
        # model.load_model(69.12, 82.47, id='big-lstm-precise', epoch_num=66)
        # model.load_model(82.94, 80.47, id='big-lstm-for-f1', epoch_num=149)
        # model.load_model(38.89, 38.22, id='cqt-no-log-256', epoch_num=11)
        # model.load_model(28.68, 9.22, id='cqt-no-log-256', epoch_num=0)
        model.build_model()
        model.load_newest(hparams.load_id)
    elif model_type == ModelType.TIMBRE:
        # model.load_model(23.81, id='96er', epoch_num=19)
        # model.load_model(24.22, id='96er', epoch_num=352)
        # model.load_model(30.56, id='96er', epoch_num=163)
        # model.load_model(12.31, id='96er', epoch_num=88)
        # model.load_model(9.51, id='96er', epoch_num=10)
        # model.load_model(6.40, id='no-bot', epoch_num=17)
        # model.load_model(15.49, id='no-bot', epoch_num=78)
        model.build_model()
        model.load_newest(hparams.load_id)
        # model.load_model(5.17, id=hparams.model_id, epoch_num=8)
    else:
        print('building full model')
        midi_model = ModelWrapper(model_dir, ModelType.MELODIC, hparams=hparams)
        midi_model.build_model(compile=False)
        midi_model.load_newest()
        timbre_model = ModelWrapper(model_dir, ModelType.TIMBRE, hparams=hparams)
        timbre_model.build_model(compile=False)
        timbre_model.load_newest()

        model.build_model(midi_model=midi_model.get_model(), timbre_model=timbre_model.get_model())

        model.load_newest(hparams.load_id)

    graph = tf.compat.v1.get_default_graph()
    graph.finalize()
    for i in range(num_steps):
        model.train_and_save(epochs=1, epoch_num=i)

    # estimator.train(input_fn=transcription_data, max_steps=num_steps)


def transcribe(data_fn,
               model_dir,
               model_type,
               path,
               file_suffix,
               hparams):
    if data_fn:
        transcription_data = data_fn(preprocess_examples=True,
                                     is_training=False,
                                     shuffle_examples=True,
                                     skip_n_initial_records=0,
                                     params=hparams)
    else:
        transcription_data = None

    if model_type == ModelType.MELODIC:
        midi_model = ModelWrapper(model_dir, ModelType.MELODIC, dataset=transcription_data,
                                  batch_size=1, id_=hparams.model_id, hparams=hparams)
        # midi_model.load_model(74.87, 82.45, 'weights-zero')
        # midi_model.load_model(82.94, 80.47, id='big-lstm-for-f1', epoch_num=149)
        midi_model.build_model(compile=False)
        midi_model.load_newest(hparams.load_id)
    elif model_type == ModelType.TIMBRE:
        timbre_model = ModelWrapper(model_dir, ModelType.TIMBRE, id_=hparams.model_id,
                                    dataset=transcription_data, batch_size=1,
                                    hparams=hparams)
        timbre_model.build_model(compile=False)
        timbre_model.load_newest(hparams.load_id)

    if data_fn:
        while True:
            if model_type == ModelType.MELODIC:
                x, _ = midi_model.generator.get()
                sequence_prediction = midi_model.predict_from_spec(x[0])
                midi_filename = path + file_suffix + '.midi'
                midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)
            elif model_type == ModelType.TIMBRE:
                x, y = timbre_model.generator.get()
                timbre_prediction = K.get_value(timbre_model.predict_from_spec(*x))[0]
                print(
                    f'True: {x[1][0][0]}{constants.FAMILY_IDX_STRINGS[np.argmax(y[0][0])]}. Predicted: {constants.FAMILY_IDX_STRINGS[timbre_prediction]}')
    else:
        filenames = glob.glob(path)

        for filename in filenames:
            wav_data = tf.io.gfile.GFile(filename, 'rb').read()

            if model_type == ModelType.MELODIC:
                spec = wav_to_spec_op(wav_data, hparams=hparams)

                # add "batch" and channel dims
                spec = tf.reshape(spec, (1, *spec.shape, 1))
                sequence_prediction = midi_model.predict_from_spec(spec)
                midi_filename = filename + file_suffix + '.midi'
                midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)
            elif model_type == ModelType.TIMBRE:
                y = audio_io.wav_data_to_samples(wav_data, hparams.sample_rate)
                spec = create_spectrogram(K.constant(y), hparams)
                # add "batch" and channel dims
                spec = K.cast_to_floatx(tf.reshape(spec, (1, *spec.shape, 1)))
                timbre_prediction = K.get_value(timbre_model.predict_from_spec(spec))[0]
                print(
                    f'File: {filename}. Predicted: {constants.FAMILY_IDX_STRINGS[timbre_prediction]}')


def evaluate(data_fn,
             additional_trial_info,
             model_dir,
             model_type,
             preprocess_examples,
             hparams,
             name,
             num_steps=None,
             note_based=False):
    """Evaluation loop."""
    hparams.batch_size = 1
    hparams.slakh_batch_size = 1

    transcription_data_base = functools.partial(
        data_fn,
        preprocess_examples=preprocess_examples,
        is_training=False)

    if True or num_steps is None:
        transcription_data = functools.partial(
            transcription_data_base,
            shuffle_examples=False, skip_n_initial_records=0)
    else:
        # If num_steps is specified, we will evaluate only a subset of the data.
        #
        # The following is a hack that works around the problems of not being able
        # to determine the number of records in a given TFRecord shard without
        # reading the whole thing and not being able to persist a tf.data.Dataset
        # session across multiple estimator evaluate calls.
        #
        # This code tries to select a different subset for every evaluation by doing
        # the following:
        # - Setting shuffle_examples=True. This shuffles not only individual
        #   examples, but also shuffles the order in which shards are read.
        # - Skipping N examples before starting evaluation, where N is selected
        #   randomly for each evaluation run. This provides a different starting
        #   offset.

        # In order to skip a random number of records, we need to provide an upper
        # bound that will still let us run num_steps evaluation steps before running
        # out of data. The following code does a one-time check on startup to see
        # if there are up to num_steps * 5 records available, which would allow
        # a maximum skip range of [0, num_steps*4].
        records_to_check = num_steps * 5
        tf.compat.v1.logging.info('Checking for at least %d records...', records_to_check)
        records_available = 0
        record_check_params = copy.deepcopy(hparams)
        record_check_params.batch_size = 1
        iterator = iter(transcription_data_base(
            params=record_check_params,
            shuffle_examples=False,
            skip_n_initial_records=1000,
        ))
        next_record = next(iterator)
        with tf.Session() as sess:
            sess.run(iterator.initializer)
            try:
                for i in range(records_to_check):
                    del i
                    sess.run(next_record)
                    records_available += 1
                    if records_available % 10 == 0:
                        tf.compat.v1.logging.info('Found %d records...', records_available)
            except tf.errors.OutOfRangeError:
                pass
        # Determine max number of records we could skip and still have num_steps
        # records remaining.
        max_records_to_skip = max(0, records_available - num_steps)
        tf.compat.v1.logging.info('Found at least %d records. '
                                  'Will skip a maximum of %d records during eval runs '
                                  'in order to support %d evaluation steps.',
                                  records_available, max_records_to_skip, num_steps)

        # Since we're doing a limited number of steps, we should shuffle the
        # examples we're evaluating so each evaluation is over a different portion
        # of the dataset.
        def transcription_data(params, *args, **kwargs):
            assert not args
            skip_n_initial_records = random.randint(0, max_records_to_skip)
            tf.compat.v1.logging.info('Skipping %d initial record(s)', skip_n_initial_records)
            return transcription_data_base(
                params=params,
                shuffle_examples=True,
                skip_n_initial_records=skip_n_initial_records,
                **kwargs)

    _trial_summary(
        hparams=hparams,
        model_dir=model_dir,
        output_dir='./out',
        additional_trial_info=additional_trial_info)

    model_wrapper = ModelWrapper(model_dir, ModelType.FULL,
                              dataset=transcription_data(params=hparams),
                              batch_size=hparams.batch_size,
                              steps_per_epoch=hparams.epochs_per_save,
                              hparams=hparams)
    if model_type is ModelType.TIMBRE:
        timbre_model = ModelWrapper(model_dir, ModelType.TIMBRE, hparams=hparams)
        timbre_model.build_model(compile=False)
        model_wrapper.build_model(compile=False, timbre_model=timbre_model.get_model())
        model_wrapper.load_newest(hparams.load_id)
        model_wrapper = timbre_model
        # model_wrapper.load_newest(hparams.load_id)
    elif model_type is ModelType.MELODIC:
        midi_model = ModelWrapper(model_dir, ModelType.MELODIC, hparams=hparams, batch_size=1)
        midi_model.build_model(compile=False)
        model_wrapper.build_model(compile=False, midi_model=midi_model.get_model())
        midi_model.load_newest(hparams.load_id)
        model_wrapper = midi_model
    else:
        model_wrapper.build_model(compile=False)
        model_wrapper.load_newest(hparams.load_id)

    generator = DataGenerator(transcription_data(params=hparams), hparams.batch_size, 1,
                              use_numpy=False,
                              coagulate_mini_batches=False)
    save_dir = f'{model_dir}/{model_type.name}/{model_wrapper.id}_eval'

    if model_type is ModelType.MELODIC:
        metrics = MidiPredictionMetrics(generator=generator, note_based=note_based, hparams=hparams, save_dir=save_dir)
    else:
        metrics = EvaluationMetrics(generator=generator, hparams=hparams, save_dir=save_dir, is_full=model_type is ModelType.FULL)
    try:
        for i in range(num_steps):
            print(f'evaluating step: {i}')
            metrics.on_epoch_end(i, model=model_wrapper.get_model())
    except:
        pass

    metric_names = ['true_positives', 'false_positives', 'false_negatives']

    if model_type is not ModelType.MELODIC:
        # instrument specific metrics
        instrument_true_positives, instrument_false_positives, instrument_false_negatives = [
            functools.reduce(
                lambda a, b: a + b,
                map(lambda x: np.array(
                    [x[constants.FAMILY_IDX_STRINGS[i]][n] for i in range(len(x.keys()))]),
                    metrics.metrics_history)) for n in metric_names
        ]

        instrument_precision = instrument_true_positives \
                               / (instrument_true_positives + instrument_false_positives + 1e-9)
        instrument_recall = instrument_true_positives \
                            / (instrument_true_positives + instrument_false_negatives + 1e-9)

        instrument_f1_score = 2 * ((instrument_precision * instrument_recall)
                                   / (instrument_precision + instrument_recall + 1e-9))

        overall_precision = np.sum(instrument_true_positives[:-1]) \
                            / np.sum(instrument_true_positives[:-1]
                                     + instrument_false_positives[:-1] + 1e-9)
        overall_recall = np.sum(instrument_true_positives[:-1]) \
                         / np.sum(instrument_true_positives[:-1]
                                  + instrument_false_negatives[:-1] + 1e-9)

        overall_f1_score = 2 * ((overall_precision * overall_recall)
                                / (overall_precision + overall_recall + 1e-9))

        for i in range(hparams.timbre_num_classes + (1 if model_type is ModelType.FULL else 0)):
            instrument = constants.FAMILY_IDX_STRINGS[i]
            print(f'{instrument}: '
                  f'P: {instrument_precision[i]}, '
                  f'R: {instrument_recall[i]}, '
                  f'F1: {instrument_f1_score[i]}, '
                  f'N: {instrument_true_positives[i] + instrument_false_negatives[i]}')
        total_support = K.sum(instrument_true_positives) + K.sum(instrument_false_negatives)
        print(f'overall: '
              f'P: {overall_precision}, '
              f'R: {overall_recall}, '
              f'F1: {overall_f1_score}, '
              f'N: {total_support}')
    elif note_based:
        macro_names = ['note_precision', 'note_recall', 'note_f1_score', 'frame_precision',
                       'frame_recall', 'frame_f1_score']
        note_precision, note_recall, note_f1, frame_precision, frame_recall, frame_f1 = [
            functools.reduce(
                lambda a, b: a + b,
                map(lambda x: [x[n]],
                    metrics.metrics_history)) for n in macro_names
        ]
        print(f'nP: {np.mean(note_precision)}, '
              f'nR: {np.mean(note_recall)}, '
              f'nF: {np.mean(note_f1)}, '
              f'fP: {np.mean(frame_precision)}, '
              f'fR: {np.mean(frame_recall)}, '
              f'fF: {np.mean(frame_f1)}, ')
    else:
        stacks = ['frames', 'onsets', 'offsets']
        # instrument-agnostic metrics

        true_positives, false_positives, false_negatives = [
            functools.reduce(
                lambda a, b: a + b,
                map(lambda x: np.array(
                    [x[stacks[i]][n] for i in range(len(x.keys()))]),
                    metrics.metrics_history)) for n in metric_names
        ]
        precision = true_positives / (true_positives + false_positives + 1e-9)
        recall = true_positives / (true_positives + false_negatives + 1e-9)

        f1_score = 2 * precision * recall / (precision + recall + 1e-9)
        support = true_positives + false_negatives
        for i in range(len(stacks)):
            stack = stacks[i]
            print(f'{stack}: '
                  f'P: {precision[i]}, '
                  f'R: {recall[i]}, '
                  f'F1: {f1_score[i]}, '
                  f'N: {support[i]}')





