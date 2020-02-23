# load and save models
import os
from enum import Enum

import numpy as np
import uuid

from magenta.models.onsets_frames_transcription.callback import Metrics

import tensorflow.compat.v1 as tf
from magenta.models.onsets_frames_transcription.loss_util import log_loss_wrapper

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
    model_save_format = '{}/Training {} Model Weights {} {:.2f} {:.2f}.hdf5'
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
        self.generator = DataGenerator(self.dataset, self.batch_size, self.steps_per_epoch,
                                       use_numpy=False)
        self.metrics = Metrics(self.generator, self.hparams)

    def save_model_with_metrics(self):
        id_tup = (self.model_dir, self.type.name, self.id,
                  self.metrics.metrics_history[-1].frames['f1_score'][0].numpy() * 100,
                  self.metrics.metrics_history[-1].onsets['f1_score'][0].numpy() * 100)
        print('Saving {} model...'.format(self.type.name))
        self.model.save_weights(ModelWrapper.model_save_format.format(*id_tup))
        np.save(ModelWrapper.history_save_format.format(*id_tup), [self.metrics.metrics_history])
        print('Model weights saved at: ' + self.model_save_format.format(*id_tup))

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
                                                  + [eval_data[5]]
            self.hist['val_frames_' + acc_name] = self.hist.get('val_frames_' + acc_name, []) \
                                                  + [eval_data[4]]

        id_tup = (self.model_dir, self.type.name, self.id,
                  self.hist['val_onsets_' + acc_name][-1] * 100,
                  self.hist['val_frames_' + acc_name][-1] * 100)
        print('Saving {} model...'.format(self.type.name))
        self.model.save_weights(ModelWrapper.model_save_format.format(*id_tup))
        np.save(ModelWrapper.history_save_format.format(*id_tup), [self.hist])
        print('Model saved at: ' + self.model_save_format.format(*id_tup))

    def train_and_save(self, epochs=1):
        if self.model is None:
            self.build_model()

        self.model.fit_generator(self.generator, steps_per_epoch=self.steps_per_epoch,
                                 epochs=epochs, workers=2, max_queue_size=8,
                                 callbacks=[self.metrics])

        self.save_model_with_metrics()

    # train on a tf.data Dataset
    def train_and_save_on_dataset(self, dataset, val_dataset, epochs=1):
        self.model.fit(dataset, validation_steps=1,
                       # validation_split=0.1 if self.batch_size > 1 else 0.0,
                       validation_data=val_dataset, epochs=epochs,
                       steps_per_epoch=self.steps_per_epoch)

    # train on numpy arrays
    def train_and_save_on_numpy(self, x_train, y_train, x_val=None, y_val=None, epochs=1):
        self.model.fit(x_train, y_train, self.batch_size,
                       # validation_split=0.1 if self.batch_size > 1 else 0.0,
                       # validation_data=None if x_val is None else (x_val, y_val),
                       epochs=epochs)

    def predict(self, input):
        return self.model.predict(input)

    def load_model(self, frames_f1, onsets_f1):
        self.build_model()
        id_tup = (self.model_dir, self.type.name, self.id, frames_f1, onsets_f1)
        if os.path.exists(ModelWrapper.model_save_format.format(*id_tup)) \
                and os.path.exists(ModelWrapper.history_save_format.format(*id_tup) + '.npy'):
            self.metrics.load_metrics(
                np.load(ModelWrapper.history_save_format.format(*id_tup) + '.npy',
                        allow_pickle=True)[0])
            print('Loading pre-trained {} model...'.format(self.type.name))
            self.model.load_weights(ModelWrapper.model_save_format.format(*id_tup))
            # self.model = load_model(ModelWrapper.model_save_format.format(*id_tup),
            #                         custom_objects={
            #                             'log_loss_wrapper': log_loss_wrapper,
            #                             'binary_accuracy_wrapper': binary_accuracy_wrapper
            #                         },
            #                         compile=True)
            # self.model.compile(Adam(self.hparams.learning_rate), 'categorical_crossentropy',
            #                    metrics=['accuracy'])
        else:
            print('Pre-trained model not found')

    def build_model(self):
        if self.type == ModelType.MIDI:
            self.model, losses, accuracies = midi_prediction_model(self.hparams)
            self.model.compile(Adam(self.hparams.learning_rate),
                               metrics=accuracies, loss=losses)
        elif self.type == ModelType.TIMBRE:
            pass
        else:  # self.type == ModelType.FULL:
            pass
