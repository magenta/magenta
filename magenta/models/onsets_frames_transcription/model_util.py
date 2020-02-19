# load and save models
import os
from enum import Enum

import numpy as np
import uuid

import tensorflow.compat.v1 as tf


FLAGS = tf.app.flags.FLAGS

if FLAGS.using_plaidml:
    # os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

    import plaidml.keras

    plaidml.keras.install_backend()
    from keras.models import load_model
    import keras.backend as K
else:
    from tensorflow.keras.models import load_model
    import tensorflow.keras.backend as K

from magenta.models.onsets_frames_transcription.data_generator import DataGenerator

from magenta.models.onsets_frames_transcription.accuracy_util import AccuracyMetric
from magenta.models.onsets_frames_transcription.midi_model import midi_prediction_model


class ModelType(Enum):
    MIDI = 'Midi',
    TIMBRE = 'Timbre',
    FULL = 'Full',

def to_lists(x, y):
    return tf.data.Dataset.from_tensor_slices(x), tf.data.Dataset.from_tensor_slices(y)


class ModelWrapper:
    model_save_format = '{}/Training {} Model {} {:.2f} {:.2f}.hdf5'
    history_save_format = '{}/Training {} History {} {:.2f} {:.2f}'

    def __init__(self, model_dir, type, id=None, batch_size=8, steps_per_epoch=100,
                 dataset=None,
                 accuracy_metric=AccuracyMetric('acc', 'accuracy'),
                 model=None, hist=None, hparams=None):
        self.model_dir = model_dir
        self.type = type
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
        self.generator = None
        if hparams.using_plaidml:
            self.convert_dataset()

    def convert_dataset(self):
        self.generator = DataGenerator(self.dataset, self.batch_size, self.steps_per_epoch)

    def save_model(self, eval_data=None):
        acc_name = 'acc'
        new_hist = self.model.history.history
        if not self.hist:
            self.hist = new_hist
        if 'onsets_accuracy' in new_hist:
            acc_name = 'accuracy'

        self.hist = {key: self.hist[key] + val for key, val in new_hist.items()}
        if eval_data:
            self.hist['val_onsets_' + acc_name] = self.hist.get('val_onsets_' + acc_name, []) \
                                                  + [eval_data[3]]
            self.hist['val_frames_' + acc_name] = self.hist.get('val_frames_' + acc_name, []) \
                                                  + [eval_data[4]]

        id_tup = (self.model_dir, self.type.name, self.id,
                  self.hist['val_onsets_' + acc_name][-1] * 100,
                  self.hist['val_frames_' + acc_name][-1] * 100)
        print('Saving {} model...'.format(self.type.name))
        self.model.save(ModelWrapper.model_save_format.format(*id_tup))
        np.save(ModelWrapper.history_save_format.format(*id_tup), [self.hist])
        print('Model saved at: ' + self.model_save_format.format(*id_tup))

    def train_and_save(self, epochs=1):
        if self.model is None:
            self.build_model()
        try:
            if False and self.hparams.using_plaidml:

                for i in range(self.steps_per_epoch * epochs):
                    next_input = next(iter(self.dataset))

                    x_train, y_train = ([t.numpy() for t in tensors] for tensors in next_input)
                    if False:#val_dataset:
                        next_val = next(iter(val_dataset))
                        x_val, y_val = ([t.numpy() for t in tensors] for tensors in next_val)

                    else:
                        x_val, y_val = (None, None)
                    print('Training on "epoch" {}'.format(i))
                    self.train_and_save_on_numpy(x_train, y_train, x_val, y_val, 1)
            elif self.hparams.using_plaidml:
                self.model.fit_generator(self.generator, steps_per_epoch=self.steps_per_epoch, epochs=epochs)

            else:
                if True:#not val_dataset:
                    val_dataset = self.dataset.shuffle(self.batch_size)
                self.train_and_save_on_dataset(self.dataset, val_dataset, epochs, self.steps_per_epoch)

            print(K.get_value(self.model.optimizer.lr))
        finally:
            # always try to save if we can
            # Do this if we get a Keyboard Interrupt
            if self.hparams.using_plaidml:
                x_val, y_val = ([t.numpy() for t in tensors] for tensors in next(iter(self.dataset)))

                self.save_model(self.model.evaluate(x_val, y_val, batch_size=self.batch_size))
            else:
                self.save_model()

    # train on a tf.data Dataset
    def train_and_save_on_dataset(self, dataset, val_dataset, epochs=1):
        self.model.fit(dataset, validation_steps=1,
                       # validation_split=0.1 if self.batch_size > 1 else 0.0,
                       validation_data=val_dataset, epochs=epochs, steps_per_epoch=self.steps_per_epoch)

    # train on numpy arrays
    def train_and_save_on_numpy(self, x_train, y_train, x_val=None, y_val=None, epochs=1):
        self.model.fit(x_train, y_train, self.batch_size,
                       # validation_split=0.1 if self.batch_size > 1 else 0.0,
                       # validation_data=None if x_val is None else (x_val, y_val),
                       epochs=epochs)

    def predict(self, input):
        return self.model.predict(input)

    def load_model(self, onsets_acc, frames_acc):
        id_tup = (self.model_dir, self.type.name, self.id, onsets_acc * 100, frames_acc * 100)
        if os.path.exists(
                ModelWrapper.model_save_format.format(*id_tup) and os.path.exists(
                    ModelWrapper.history_save_format.format(*id_tup) + '.npy')):
            self.hist = \
                np.load(ModelWrapper.history_save_format.format(*id_tup) + '.npy',
                        allow_pickle=True)[0]
        print('Loading pre-trained {} model...'.format(self.type.name))
        self.model = load_model(ModelWrapper.model_save_format.format(*id_tup),
                                compile=False)
        self.model.compile('Adam', 'categorical_crossentropy', metrics=['accuracy'])

    def build_model(self):
        if self.type == ModelType.MIDI:
            self.model = midi_prediction_model(self.hparams)
            self.model.compile('Adam', 'categorical_crossentropy', metrics=['accuracy'])
        elif self.type == ModelType.TIMBRE:
            pass
        else:  # self.type == ModelType.FULL:
            pass
