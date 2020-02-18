# load and save models
import os
from enum import Enum

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import plaidml.keras

plaidml.keras.install_backend()

import tensorflow as tf
import numpy as np
from keras.models import load_model, Model
from magenta.models.onsets_frames_transcription.accuracy_util import AccuracyMetric, Dixon
from magenta.models.onsets_frames_transcription.midi_model import midi_prediction_model


class ModelType(Enum):
    MIDI = 'Midi',
    TIMBRE = 'Timbre',
    FULL = 'Full',


class ModelWrapper:
    model_save_format = '{}/Training {} Model {:.2f} {:.2f}.hdf5'
    history_save_format = '{}/Training {} History {:.2f} {:.2f}'

    def __init__(self, model_dir, type, batch_size=8,
                 accuracy_metric=AccuracyMetric('acc', 'accuracy'),
                 model=None, hist=None, hparams=None):
        self.model_dir = model_dir
        self.type = type
        self.batch_size = batch_size
        self.accuracy_metric = accuracy_metric
        self.model = model
        self.hist = hist
        self.hparams = hparams

    def save_model(self):
        new_hist = self.model.history.history
        if not self.hist:
            self.hist = new_hist
        self.hist = {key: self.hist[key] + val for key, val in new_hist.items()}
        id_tup = (self.model_dir, self.type.name,
                  self.hist['onsets_' + self.accuracy_metric.name][-1] * 100,
                  self.hist['frames_' + self.accuracy_metric.name][-1] * 100)
        self.model.save(ModelWrapper.model_save_format.format(*id_tup))
        np.save(ModelWrapper.history_save_format.format(*id_tup), [self.hist])

    def train_and_save(self, dataset, val_dataset, epochs=1):
        if self.hparams.using_plaidml:
            next_input = next(iter(dataset))

            x_train, y_train = ([t.numpy() for t in tensors] for tensors in next_input)
            if val_dataset:
                next_val = next(iter(val_dataset))
                x_val, y_val = ([t.numpy() for t in tensors] for tensors in next_val)

            else:
                x_val, y_val = None, None
            self.train_and_save_on_numpy(x_train, y_train, x_val, y_val, epochs)
        else:
            self.train_and_save_on_dataset(dataset, val_dataset, epochs)

    # train on a tf.data Dataset
    def train_and_save_on_dataset(self, dataset, val_dataset, epochs=1):
        if self.model is None:
            self.build_model()
        self.model.fit(dataset, validation_data=val_dataset, epochs=epochs, steps_per_epoch=1)
        # TODO work on saving
        self.save_model()
    # train on numpy arrays
    def train_and_save_on_numpy(self, x_train, y_train, x_val=None, y_val=None, epochs=1):
        if self.model is None:
            self.build_model()
        self.model.fit(x_train, y_train, self.batch_size,
                       validation_data=None if x_val is None else (x_val, y_val), epochs=epochs)
        # TODO work on saving
        self.save_model()

    def predict(self, input):
        return self.model.predict(input)

    def load_model(self, onsets_acc, frames_acc):
        id_tup = (self.model_dir, self.type.name, onsets_acc * 100, frames_acc * 100)
        if os.path.exists(
                ModelWrapper.model_save_format.format(*id_tup) and os.path.exists(
                    ModelWrapper.history_save_format.format(*id_tup))):
            self.hist = \
                np.load(ModelWrapper.history_save_format.format(*id_tup), allow_pickle=True)[0]
        print('Loading pre-trained {} model...'.format(self.type.name))
        self.model = load_model(ModelWrapper.model_save_format.format(*id_tup),
                                {self.accuracy_metric.name: self.accuracy_metric.method})

    def build_model(self):
        if self.type == ModelType.MIDI:
            self.model = midi_prediction_model(self.hparams)
            self.model.compile('Adam', 'categorical_crossentropy', metrics=['accuracy'])
        elif self.type == ModelType.TIMBRE:
            pass
        else:  # self.type == ModelType.FULL:
            pass
