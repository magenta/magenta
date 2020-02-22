import collections

import numpy as np
from keras.callbacks import Callback
from magenta.models.onsets_frames_transcription.metrics import define_metrics, \
    calculate_frame_metrics
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

OutputMetrics = collections.namedtuple('OutputMetrics', ('frames', 'onsets', 'offsets'))


class Metrics(Callback):
    def __init__(self, generator=None, hparams=None, metrics_history=None):
        super(Metrics, self).__init__()
        if metrics_history is None:
            metrics_history = []
        self.generator = generator
        self.hparams = hparams
        self.metrics_history = metrics_history

    def on_train_batch_begin(self, *args):
        pass

    def on_train_batch_end(self, *args):
        pass

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

        return OutputMetrics(frame_metrics, onset_metrics, offset_metrics)

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.generator.get()
        metrics = self.predict(x, y)
        self.metrics_history.append(metrics)
        for name, value in metrics._asdict().items():
            print('{} metrics:'.format(name))
            print('Precision: {}, Recall: {}, F1: {}\n'.format(value['precision'][0].numpy(),
                                                               value['recall'][0].numpy(),
                                                               value['f1_score'][0].numpy()))
