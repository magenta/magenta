# load and save models
import os
from enum import Enum

import librosa.display
import numpy as np
import uuid
import matplotlib.pyplot as plt

from magenta.models.onsets_frames_transcription import infer_util, constants
from magenta.models.onsets_frames_transcription.callback import MidiPredictionMetrics, \
    TimbrePredictionMetrics

import tensorflow.compat.v1 as tf
from magenta.models.onsets_frames_transcription.loss_util import log_loss_wrapper
from magenta.models.onsets_frames_transcription.timbre_model import timbre_prediction_model, \
    high_pass_filter

FLAGS = tf.app.flags.FLAGS

if FLAGS.using_plaidml:
    # os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

    import plaidml.keras

    plaidml.keras.install_backend()
    from keras.models import load_model
    from keras.optimizers import Adam
    import keras.backend as K
else:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam
    import tensorflow.keras.backend as K

from magenta.models.onsets_frames_transcription.data_generator import DataGenerator

from magenta.models.onsets_frames_transcription.accuracy_util import AccuracyMetric, \
    binary_accuracy_wrapper
from magenta.models.onsets_frames_transcription.midi_model import midi_prediction_model


class ModelType(Enum):
    MIDI = 'Midi',
    TIMBRE = 'Timbre',
    FULL = 'Full',


def to_lists(x, y):
    return tf.data.Dataset.from_tensor_slices(x), tf.data.Dataset.from_tensor_slices(y)


class ModelWrapper:
    def __init__(self, model_dir, type, id=None, batch_size=8, steps_per_epoch=100,
                 dataset=None,
                 accuracy_metric=AccuracyMetric('acc', 'accuracy'),
                 model=None, hist=None, hparams=None):
        self.model_dir = model_dir
        self.type = type

        self.model_save_format = '{}/Training {} Model Weights {} {:.2f} {:.2f}.hdf5' \
            if type is ModelType.MIDI \
            else '{}/Training {} Model Weights {} {:.2f}.hdf5'
        self.history_save_format = '{}/Training {} History {} {:.2f} {:.2f}' \
            if type is ModelType.MIDI \
            else '{}/Training {} History {} {:.2f}.hdf5'
        if id is None:
            self.id = uuid.uuid4().hex
        else:
            self.id = id
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.accuracy_metric = accuracy_metric
        self.model = model
        self.hist = hist
        self.hparams = hparams

        self.dataset = dataset
        if dataset is None:
            self.generator = None
        else:
            self.generator = DataGenerator(self.dataset, self.batch_size, self.steps_per_epoch,
                                           use_numpy=True)
        self.metrics = MidiPredictionMetrics(self.generator, self.hparams) if type == ModelType.MIDI else TimbrePredictionMetrics(self.generator, self.hparams)

    def save_model_with_metrics(self):
        if self.type == ModelType.MIDI:
            id_tup = (self.model_dir, self.type.name, self.id,
                      self.metrics.metrics_history[-1].frames['f1_score'].numpy() * 100,
                      self.metrics.metrics_history[-1].onsets['f1_score'].numpy() * 100)
        else:
            id_tup = (self.model_dir, self.type.name, self.id,
                      self.metrics.metrics_history[-1].timbre_prediction['f1_score'].numpy() * 100)
        print('Saving {} model...'.format(self.type.name))
        self.model.save_weights(self.model_save_format.format(*id_tup))
        np.save(self.history_save_format.format(*id_tup), [self.metrics.metrics_history])
        print('Model weights saved at: ' + self.model_save_format.format(*id_tup))

    def train_and_save(self, epochs=1):
        if self.model is None:
            self.build_model()

        # self.model.fit_generator(self.generator, steps_per_epoch=self.steps_per_epoch,
        #                          epochs=epochs, workers=2, max_queue_size=8,
        #                          callbacks=[self.metrics])
        # self.model.fit(self.dataset, steps_per_epoch=self.steps_per_epoch,
        #                epochs=epochs, workers=2, max_queue_size=8,
        #                callbacks=[self.metrics])
        for i in range(self.steps_per_epoch):
            x, y = self.generator.get()
            print(np.argmax(y[0], 1))
            '''print(x[1])
            print('family: {}'.format(y[0]))
            foo = high_pass_filter(x)
            plt.figure(figsize=(12,8))
            plt.subplot(2, 1, 1)
            librosa.display.specshow(librosa.power_to_db(tf.transpose(tf.squeeze(x[0][0])).numpy()),
                                     y_axis='cqt_note',
                                     hop_length=self.hparams.timbre_hop_length,
                                     fmin=constants.MIN_TIMBRE_PITCH,
                                     bins_per_octave=constants.BINS_PER_OCTAVE)
            plt.subplot(2, 1, 2)
            librosa.display.specshow(librosa.power_to_db(tf.transpose(tf.squeeze(foo[0])).numpy()),
                                     y_axis='cqt_note',
                                     hop_length=self.hparams.timbre_hop_length,
                                     fmin=constants.MIN_TIMBRE_PITCH,
                                     bins_per_octave=constants.BINS_PER_OCTAVE)
            plt.show()'''
            new_metrics = self.model.train_on_batch(x, y)
            print(new_metrics)
        self.metrics.on_epoch_end(1, model=self.model)

        self.save_model_with_metrics()

    def predict_sequence(self, input):

        y_pred = self.model.predict(input)
        frame_predictions = y_pred[0][0] > self.hparams.predict_frame_threshold
        onset_predictions = y_pred[1][0] > self.hparams.predict_onset_threshold
        offset_predictions = y_pred[2][0] > self.hparams.predict_offset_threshold

        # frame_predictions = tf.expand_dims(frame_predictions, axis=0)
        # onset_predictions = tf.expand_dims(onset_predictions, axis=0)
        # offset_predictions = tf.expand_dims(offset_predictions, axis=0)
        sequence = infer_util.predict_sequence(
                frame_predictions=frame_predictions,
                onset_predictions=onset_predictions,
                offset_predictions=offset_predictions,
                velocity_values=None,
                hparams=self.hparams, min_pitch=constants.MIN_MIDI_PITCH)
        return sequence



    def load_model(self, frames_f1, onsets_f1=-1, id=-1):
        if not id:
            id = self.id
        self.build_model()
        if self.type == ModelType.MIDI:
            id_tup = (self.model_dir, self.type.name, id, frames_f1, onsets_f1)
        else:
            id_tup = (self.model_dir, self.type.name, id, frames_f1)
        if os.path.exists(self.model_save_format.format(*id_tup)) \
                and os.path.exists(self.history_save_format.format(*id_tup) + '.npy'):
            self.metrics.load_metrics(
                np.load(self.history_save_format.format(*id_tup) + '.npy',
                        allow_pickle=True)[0])
            print('Loading pre-trained {} model...'.format(self.type.name))
            self.model.load_weights(self.model_save_format.format(*id_tup))
        else:
            print('Couldn\'t find pre-trained model: {}'
                  .format(self.model_save_format.format(*id_tup)))

    # num_classes only needed for timbre prediction
    def build_model(self):
        if self.type == ModelType.MIDI:
            self.model, losses, accuracies = midi_prediction_model(self.hparams)
            self.model.compile(Adam(self.hparams.learning_rate),
                               metrics=accuracies, loss=losses)
        elif self.type == ModelType.TIMBRE:
            self.model, losses, accuracies = timbre_prediction_model(self.hparams)
            self.model.compile(Adam(self.hparams.learning_rate),
                               metrics=accuracies, loss=losses)
        else:  # self.type == ModelType.FULL:
            pass
