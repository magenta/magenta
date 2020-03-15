import collections
from abc import abstractmethod

import tensorflow.compat.v1 as tf
import tensorflow.keras.backend as K

from magenta.models.onsets_frames_transcription.accuracy_util import flatten_f1_wrapper, \
    multi_track_accuracy_wrapper

FLAGS = tf.app.flags.FLAGS

if FLAGS.using_plaidml:
    from keras.callbacks import Callback
else:
    from tensorflow.keras.callbacks import Callback

from magenta.models.onsets_frames_transcription.metrics import calculate_frame_metrics

MidiPredictionOutputMetrics = collections.namedtuple('MidiPredictionOutputMetrics',
                                                     ('frames', 'onsets', 'offsets'))
TimbrePredictionOutputMetrics = collections.namedtuple('TimbrePredictionOutputMetrics',
                                                       ('timbre_prediction',))


class MetricsCallback(Callback):
    def __init__(self, generator=None, hparams=None, metrics_history=None):
        super(MetricsCallback, self).__init__()
        if metrics_history is None:
            metrics_history = []
        self.generator = generator
        self.hparams = hparams
        self.metrics_history = metrics_history

    def on_train_batch_begin(self, *args):
        pass

    def on_train_batch_end(self, *args):
        pass

    @abstractmethod
    def predict(self, X, y):
        pass

    def on_epoch_end(self, epoch, logs={}, model=None):
        if model:
            self.model = model
        x, y = self.generator.get()
        metrics = self.predict(x, y)
        self.metrics_history.append(metrics)
        for name, value in metrics._asdict().items():
            print('{} metrics:'.format(name))
            print('Precision: {}, Recall: {}, F1: {}\n'.format(value['precision'].numpy() * 100,
                                                               value['recall'].numpy() * 100,
                                                               value['f1_score'].numpy() * 100))


class MidiPredictionMetrics(MetricsCallback):
    def load_metrics(self, metrics_history):
        # convert to list of namedtuples
        self.metrics_history = [MidiPredictionOutputMetrics(*x) for x in metrics_history]
    def predict(self, X, y):
        # 'frames': boolean_accuracy_wrapper(hparams.predict_frame_threshold),
        # 'onsets': boolean_accuracy_wrapper(hparams.predict_onset_threshold),
        # 'offsets': boolean_accuracy_wrapper(hparams.predict_offset_threshold)
        y_probs = self.model.predict_on_batch(X)
        frame_metrics = calculate_frame_metrics(y[0],
                                                y_probs[0] > self.hparams.predict_frame_threshold)
        onset_metrics = calculate_frame_metrics(y[1],
                                                y_probs[1] > self.hparams.predict_onset_threshold)
        offset_metrics = calculate_frame_metrics(y[2],
                                                 y_probs[2] > self.hparams.predict_offset_threshold)

        return MidiPredictionOutputMetrics(frame_metrics, onset_metrics, offset_metrics)


class TimbrePredictionMetrics(MetricsCallback):

    def load_metrics(self, metrics_history):
        # convert to list of namedtuples
        self.metrics_history = [TimbrePredictionOutputMetrics(*x) for x in metrics_history]

    def predict(self, X, y):
        y_probs = self.model.predict_on_batch(X)
        print(y_probs + K.cast_to_floatx(y[0]))
        scores = flatten_f1_wrapper(self.hparams)(y[0], y_probs)
        return TimbrePredictionOutputMetrics(scores)

class FullPredictionMetrics(MetricsCallback):
    def load_metrics(self, metrics_history):
        # convert to list of namedtuples
        self.metrics_history = [MidiPredictionOutputMetrics(*x) for x in metrics_history]

    def predict(self, X, y):
        y_probs = self.model.predict_on_batch(X)
        frame_metrics = multi_track_accuracy_wrapper(self.hparams.predict_frame_threshold)(y[0], y_probs[0])
        onset_metrics = multi_track_accuracy_wrapper(self.hparams.predict_onset_threshold)(y[1], y_probs[1])
        offset_metrics = multi_track_accuracy_wrapper(self.hparams.predict_offset_threshold)(y[2], y_probs[2])
        return MidiPredictionOutputMetrics(frame_metrics, onset_metrics, offset_metrics)
