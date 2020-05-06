# Copyright 2020 Jack Spencer Smith.
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

"""Run, Load, and save models."""
import glob
import os
import time
import uuid
from enum import Enum

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from magenta.models.polyamp import constants, infer_util
from magenta.models.polyamp.accuracy_util import convert_multi_instrument_probs_to_predictions, \
    multi_track_prf_wrapper
from magenta.models.polyamp.callback import FullPredictionMetrics, \
    MidiPredictionMetrics, TimbrePredictionMetrics
from magenta.models.polyamp.data_generator import DataGenerator
from magenta.models.polyamp.full_model import FullModel
from magenta.models.polyamp.layer_util import get_croppings_for_single_image
from magenta.models.polyamp.melodic_model import get_melodic_model
from magenta.models.polyamp.timbre_dataset_reader import NoteCropping
from magenta.models.polyamp.timbre_model import timbre_prediction_model


class ModelType(Enum):
    MELODIC = 'melodic'
    TIMBRE = 'timbre'
    FULL = 'full'


def to_lists(x, y):
    return tf.data.Dataset.from_tensor_slices(x), tf.data.Dataset.from_tensor_slices(y)


class ModelWrapper:
    def __init__(self, model_dir, type_, id_=None, batch_size=8, steps_per_epoch=100,
                 dataset=None, model=None, hist=None, hparams=None):
        self.model_dir = model_dir
        self.type = type_

        self.model_save_format = '{}/{}/{}/Training Model Weights {:.2f} {:.2f} {}.hdf5' \
            if type_ is not ModelType.TIMBRE \
            else '{}/{}/{}/Training Model Weights {:.2f} {}.hdf5'
        self.history_save_format = '{}/{}/{}/Training History {:.2f} {:.2f} {}' \
            if type_ is not ModelType.TIMBRE \
            else '{}/{}/{}/Training History {:.2f} {}.hdf5'
        if id_ is None:
            self.id = uuid.uuid4().hex
        else:
            self.id = id_
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.model: Model = model
        self.hist = hist
        self.hparams = hparams

        self.dataset = dataset
        if dataset is None:
            self.generator = None
        else:
            self.generator = DataGenerator(self.dataset, self.batch_size, self.steps_per_epoch,
                                           use_numpy=False)
        save_dir = f'{self.model_dir}/{self.type.name}/{self.id}'
        if self.type is ModelType.MELODIC:
            self.metrics = MidiPredictionMetrics(self.generator, self.hparams, save_dir=save_dir)
        elif self.type is ModelType.TIMBRE:
            self.metrics = TimbrePredictionMetrics(self.generator, self.hparams)
        else:
            self.metrics = FullPredictionMetrics(self.generator, self.hparams, save_dir=save_dir)

    def get_model(self):
        return self.model

    def save_model_with_metrics(self, epoch_num):
        if self.type == ModelType.MELODIC or self.type == ModelType.FULL:
            id_tup = (self.model_dir, self.type.name, self.id,
                      self.metrics.metrics_history[-1].frames['f1_score'].numpy() * 100,
                      self.metrics.metrics_history[-1].onsets['f1_score'].numpy() * 100,
                      epoch_num)
        else:
            id_tup = (self.model_dir, self.type.name, self.id,
                      self.metrics.metrics_history[-1].timbre_prediction['f1_score'].numpy() * 100,
                      epoch_num)
        print('Saving {} model...'.format(self.type.name))

        if not os.path.exists(f'{self.model_dir}/{self.type.name}/{self.id}'):
            os.makedirs(f'{self.model_dir}/{self.type.name}/{self.id}')
        self.model.save_weights(self.model_save_format.format(*id_tup))
        np.save(self.history_save_format.format(*id_tup), [self.metrics.metrics_history])
        print('Model weights saved at: ' + self.model_save_format.format(*id_tup))

    def train_and_save(self, epochs=1, epoch_num=0):
        if self.model is None:
            self.build_model()

        if self.steps_per_epoch <= 0:
            steps = epoch_num - self.steps_per_epoch
        else:
            steps = self.steps_per_epoch

        for i in range(steps):
            x, y = self.generator.get()

            start = time.perf_counter()
            # new_metrics = self.model.predict(x)

            new_metrics = self.model.train_on_batch(x, y)
            # tf.random.set_random_seed(1)
            # self.model.evaluate(x, y)
            # print(self.model.trainable_weights[-1])

            print(f'Trained batch {i} in {time.perf_counter() - start:0.4f} seconds')
            print([f'{x[0]}: {x[1]:0.4f}' for x in zip(self.model.metrics_names, new_metrics)])

        self.metrics.on_epoch_end(epoch_num, model=self.model)

        self.save_model_with_metrics(epoch_num)

        # try to prevent oom
        # del self.model
        # gc.collect()
        # tf.reset_default_graph()
        # K.clear_session()
        # self.build_model()
        # self.load_newest()

    def plot_spectrograms(self, x, temporal_ds=16, freq_ds=4, max_batches=1):
        for batch_idx in range(max_batches):
            spec = K.pool2d(x[0], (temporal_ds, freq_ds), (temporal_ds, freq_ds), padding='same')
            croppings = get_croppings_for_single_image(spec[batch_idx], x[1][batch_idx],
                                                       self.hparams, temporal_ds)
            plt.figure(figsize=(10, 6))
            num_crops = 0  # min(3, K.int_shape(x[1][batch_idx])[0])
            # plt.subplot(int(num_crops / 2 + 1), 2, 1)
            plt.subplot(1, 1, 1)
            y_axis = 'cqt_note' if self.hparams.timbre_spec_type == 'cqt' else 'mel'
            librosa.display.specshow(self.spec_to_db(x[0], batch_idx),
                                     y_axis=y_axis,
                                     hop_length=self.hparams.timbre_hop_length,
                                     fmin=librosa.midi_to_hz(constants.MIN_TIMBRE_PITCH),
                                     fmax=librosa.midi_to_hz(constants.MAX_TIMBRE_PITCH),
                                     bins_per_octave=constants.BINS_PER_OCTAVE)
            for i in range(num_crops):
                plt.subplot(int(num_crops / 2 + 1), 2, i + 2)
                db = self.spec_to_db(K.expand_dims(croppings, 1), i)
                librosa.display.specshow(db,
                                         y_axis=y_axis,
                                         hop_length=self.hparams.timbre_hop_length,
                                         fmin=librosa.midi_to_hz(constants.MIN_TIMBRE_PITCH),
                                         fmax=librosa.midi_to_hz(constants.MAX_TIMBRE_PITCH),
                                         bins_per_octave=constants.BINS_PER_OCTAVE / freq_ds)
        plt.show()

    def spec_to_db(self, spec_batch, i):
        if self.hparams.timbre_spec_log_amplitude:
            db = K.permute_dimensions(K.reshape(spec_batch[i] * 80,
                                                (spec_batch[i].shape[0],
                                                 spec_batch[i].shape[1],
                                                 -1)),
                                      (2, 1, 0))[-1].numpy() + \
                 librosa.power_to_db(np.array([1e-9]))[0]
        else:
            db = librosa.power_to_db(
                tf.transpose(tf.reshape(spec_batch[i], spec_batch[i].shape[0:-1])).numpy())
        return db

    def predict_next(self):
        x, _ = self.generator.get()
        return self.predict_from_spec(*x)

    def _predict_timbre(self, spec, note_croppings=None, num_notes=None):
        if note_croppings is None:  # or num_notes is None:
            pitch = librosa.note_to_midi('C3')
            start_idx = 0
            end_idx = self.hparams.timbre_hop_length * spec.shape[1]
            note_croppings = [NoteCropping(pitch=pitch,
                                           start_idx=start_idx,
                                           end_idx=end_idx)]
            note_croppings = tf.reshape(note_croppings, (1, 1, 3))
            # num_notes = tf.expand_dims(1, axis=0)

        self.plot_spectrograms([spec, note_croppings])

        timbre_probs = self.model.predict([spec, note_croppings])
        print(timbre_probs)
        return K.flatten(tf.nn.top_k(timbre_probs).indices)

    def _predict_sequence(self, spec, qpm=None):
        y_pred = self.model.predict(spec)
        frame_predictions = y_pred[0][0] > self.hparams.predict_frame_threshold
        onset_predictions = y_pred[1][0] > self.hparams.predict_onset_threshold
        active_onsets = y_pred[1][0] > self.hparams.active_onset_threshold
        offset_predictions = y_pred[2][0] > self.hparams.predict_offset_threshold

        # frame_predictions = tf.expand_dims(frame_predictions, axis=0)
        # onset_predictions = tf.expand_dims(onset_predictions, axis=0)
        # offset_predictions = tf.expand_dims(offset_predictions, axis=0)
        sequence = infer_util.predict_sequence(
            frame_predictions=frame_predictions,
            onset_predictions=onset_predictions,
            offset_predictions=offset_predictions,
            active_onsets=active_onsets,
            velocity_values=None,
            hparams=self.hparams,
            min_pitch=constants.MIN_MIDI_PITCH,
            qpm=qpm)
        return sequence

    # Stairway to heaven transcription times
    # no split: 11:06
    # duration=16: 01:46
    # duration=32: 02:07
    # duration=8: 01:40
    def _split_and_predict(self, midi_spec, timbre_spec, present_instruments, duration=16):
        samples_length = duration * self.hparams.sample_rate
        frames, onsets, offsets = None, None, None
        midi_spec_len = K.int_shape(midi_spec)[1]
        timbre_spec_len = K.int_shape(timbre_spec)[1]
        edge_spacing = 16
        for i in range(0, K.int_shape(midi_spec)[1] * self.hparams.spec_hop_length, samples_length):
            m_start = int(i / self.hparams.spec_hop_length)
            m_end = min(midi_spec_len,
                        int(edge_spacing * 2 + (i + samples_length) / self.hparams.spec_hop_length))
            t_start = int(i / self.hparams.timbre_hop_length)
            t_end = min(timbre_spec_len, int(
                edge_spacing * 4 + (i + samples_length) / self.hparams.timbre_hop_length))
            split_pred = self.model.call([
                K.expand_dims(midi_spec[0, m_start:m_end], 0),
                K.expand_dims(timbre_spec[0, t_start:t_end], 0),
                K.cast_to_floatx(present_instruments)],
                training=False)
            if i == 0:
                frames = split_pred[0][0][:-edge_spacing]
                onsets = split_pred[1][0][:-edge_spacing]
                offsets = split_pred[2][0][:-edge_spacing]
            else:
                # ignore the edge temoral info
                frames = np.concatenate([frames, split_pred[0][0][edge_spacing:-edge_spacing]],
                                        axis=0)
                onsets = np.concatenate([onsets, split_pred[1][0][edge_spacing:-edge_spacing]],
                                        axis=0)
                offsets = np.concatenate([offsets, split_pred[2][0][edge_spacing:-edge_spacing]],
                                         axis=0)
        return [np.expand_dims(frames, 0),
                np.expand_dims(onsets, 0),
                np.expand_dims(offsets, 0)]

    def predict_multi_sequence(self, midi_spec, timbre_spec, present_instruments=None, qpm=None):
        if present_instruments is None:
            present_instruments = K.expand_dims(np.ones(self.hparams.timbre_num_classes), 0)
        # y_pred = self.model.predict_on_batch([midi_spec, timbre_spec, present_instruments])
        y_pred = self._split_and_predict(midi_spec, timbre_spec, present_instruments)
        multi_track_prf_wrapper(threshold=self.hparams.predict_frame_threshold,
                                multiple_instruments_threshold=self.hparams.multiple_instruments_threshold,
                                hparams=self.hparams, print_report=True, only_f1=False)(
            K.cast_to_floatx(y_pred[1] > self.hparams.predict_frame_threshold), y_pred[1])
        permuted_y_probs = K.permute_dimensions(y_pred[1][0], (2, 0, 1))
        print(
            f'total mean: {[f"{i}:{K.max(permuted_y_probs[i])}" for i, x in enumerate(permuted_y_probs)]}')

        # self.model.train_on_batch([midi_spec, timbre_spec, present_instruments], [K.cast_to_floatx(y > 0.5) for y in y_pred])

        frame_predictions = convert_multi_instrument_probs_to_predictions(
            y_pred[0],
            self.hparams.predict_frame_threshold,
            self.hparams.multiple_instruments_threshold)[0]
        onset_predictions = convert_multi_instrument_probs_to_predictions(
            y_pred[1],
            self.hparams.predict_onset_threshold,
            self.hparams.multiple_instruments_threshold)[0]
        offset_predictions = convert_multi_instrument_probs_to_predictions(
            y_pred[2],
            self.hparams.predict_offset_threshold,
            self.hparams.multiple_instruments_threshold)[0]
        active_onsets = convert_multi_instrument_probs_to_predictions(
            y_pred[1],
            self.hparams.active_onset_threshold,
            self.hparams.multiple_instruments_threshold)[0]

        if self.hparams.use_all_instruments:
            """
            mute the instruments we don't want here
            if this happens we are trying to isolate certain instruments,
            knowing there may be others
            otherwise, the present_instruments represent the instruments that we think exist in
            the track
            For Example: A guitar-only track would set use_all_instruments to false, and set guitar
                to the only present instrument.
                        A multi-instrument track could set use_all_instruments to true, and set
                guitar to the only present instrument.
            """
            frame_predictions = tf.logical_and(frame_predictions, present_instruments > 0)
            onset_predictions = tf.logical_and(onset_predictions, present_instruments > 0)
            offset_predictions = tf.logical_and(offset_predictions, present_instruments > 0)
            active_onsets = tf.logical_and(active_onsets, present_instruments > 0)

        return infer_util.predict_multi_sequence(frame_predictions, onset_predictions,
                                                 offset_predictions, active_onsets, qpm=qpm,
                                                 hparams=self.hparams,
                                                 min_pitch=constants.MIN_MIDI_PITCH)

    def predict_from_spec(self, spec=None, num_croppings=None, additional_spec=None, num_notes=None,
                          qpm=None,
                          *args):
        if self.type == ModelType.MELODIC:
            return self._predict_sequence(spec, qpm=qpm)
        elif self.type == ModelType.TIMBRE:
            return self._predict_timbre(spec, num_croppings)
        else:
            return self.predict_multi_sequence(midi_spec=spec, timbre_spec=additional_spec)

    def load_newest(self, id='*'):
        try:
            model_weights = \
                sorted(glob.glob(
                    f'{self.model_dir}/{self.type.name}/{id}/Training Model Weights *.hdf5'),
                    key=os.path.getmtime)[-1]
            model_history = \
                sorted(glob.glob(f'{self.model_dir}/{self.type.name}/{id}/Training History *.npy'),
                       key=os.path.getmtime)[-1]
            self.metrics.load_metrics(
                np.load(model_history,
                        allow_pickle=True)[0])
            print('Loading pre-trained model: {}'.format(model_weights))
            self.model.load_weights(model_weights,
                                    by_name=True or self.type is not ModelType.FULL,
                                    skip_mismatch=True or self.type is not ModelType.FULL)
            print('Model loaded successfully')
        except IndexError:
            print(f'No saved models exist at path: {self.model_dir}/{self.type.name}/{id}/')
        except Exception as e:
            print(f'{e}\nCouldn\'t load model at path: {self.model_dir}/{self.type.name}/{id}/')
            raise e

    def load_model(self, frames_f1, onsets_f1=-1, id=-1, epoch_num=0):
        if not id:
            id = self.id
        self.build_model()
        if self.type == ModelType.MELODIC:
            id_tup = (self.model_dir, self.type.name, id, frames_f1, onsets_f1, epoch_num)
        else:
            id_tup = (self.model_dir, self.type.name, id, frames_f1, epoch_num)
        if os.path.exists(self.model_save_format.format(*id_tup)) \
                and os.path.exists(self.history_save_format.format(*id_tup) + '.npy'):
            try:
                self.metrics.load_metrics(
                    np.load(self.history_save_format.format(*id_tup) + '.npy',
                            allow_pickle=True)[0])
                print('Loading pre-trained {} model...'.format(self.type.name))
                self.model.load_weights(self.model_save_format.format(*id_tup))
                print('Model loaded successfully')
            except:
                print(f'Couldn\'t load model weights')
        else:
            print('Couldn\'t find pre-trained model: {}'
                  .format(self.model_save_format.format(*id_tup)))

    def build_model(self, midi_model=None, timbre_model=None, compile=True):
        if self.type == ModelType.MELODIC or self.type == ModelType.FULL and midi_model is None:
            midi_model, losses, accuracies = get_melodic_model(self.hparams)

            if self.type == ModelType.MELODIC:
                if compile:
                    midi_model.compile(Adam(self.hparams.learning_rate,
                                            decay=self.hparams.decay_rate,
                                            clipnorm=self.hparams.clip_norm),
                                       metrics=accuracies, loss=losses)
                self.model = midi_model

        if self.type == ModelType.TIMBRE or self.type == ModelType.FULL and timbre_model is None:
            timbre_model, losses, accuracies = timbre_prediction_model(self.hparams)

            if self.type == ModelType.TIMBRE:
                if compile:
                    timbre_model.compile(Adam(self.hparams.timbre_learning_rate,
                                              decay=self.hparams.timbre_decay_rate,
                                              clipnorm=self.hparams.timbre_clip_norm),
                                         metrics=accuracies, loss=losses)
                self.model = timbre_model
        if self.type == ModelType.FULL:  # self.type == ModelType.FULL:
            self.model, losses, accuracies = FullModel(midi_model, timbre_model,
                                                       self.hparams).get_model()
            self.model.compile(Adam(self.hparams.full_learning_rate,
                                    # decay=self.hparams.timbre_decay_rate,
                                    # clipnorm=self.hparams.timbre_clip_norm
                                    ),
                               metrics=accuracies, loss=losses)
            # self.model.compile(optimizers.Adadelta(),
            #                    metrics=accuracies, loss=losses)

        # only save the model image if we are training on it
        if compile:
            if not os.path.exists(f'{self.model_dir}/{self.type.name}/{self.id}'):
                os.makedirs(f'{self.model_dir}/{self.type.name}/{self.id}')
            plot_model(self.model,
                       to_file=f'{self.model_dir}/{self.type.name}/{self.id}/model.png',
                       show_shapes=True,
                       show_layer_names=False)
        try:
            print(self.model.summary())
        except Exception as e:
            print(e)
