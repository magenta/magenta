import collections
import gc
import os
from abc import abstractmethod

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback

from magenta.models.polyamp import constants, infer_util
from magenta.models.polyamp.accuracy_util import \
    convert_multi_instrument_probs_to_predictions, flatten_f1_wrapper, get_last_channel, \
    multi_track_prf_wrapper, calculate_frame_metrics
from magenta.models.polyamp.metrics import calculate_metrics
from magenta.music import midi_io

MidiPredictionOutputMetrics = (
    collections.namedtuple('MidiPredictionOutputMetrics',
                           ('frames', 'onsets', 'offsets'))
)
TimbrePredictionOutputMetrics = (
    collections.namedtuple('TimbrePredictionOutputMetrics',
                           ('timbre_prediction',))
)


class MetricsCallback(Callback):
    def __init__(self,
                 generator=None,
                 hparams=None,
                 metrics_history=None,
                 save_dir='./out',
                 note_based=False):
        super(MetricsCallback, self).__init__()
        if metrics_history is None:
            metrics_history = []
        self.generator = generator
        self.hparams = hparams
        self.metrics_history = metrics_history
        self.save_dir = save_dir
        self.note_based = note_based

    def save_midi(self, y_probs, y_true, epoch):
        frame_predictions = y_probs[0][0] > self.hparams.predict_frame_threshold
        onset_predictions = y_probs[1][0] > self.hparams.predict_onset_threshold
        offset_predictions = y_probs[2][0] > self.hparams.predict_offset_threshold
        active_onsets = y_probs[1][0] > self.hparams.active_onset_threshold

        sequence = infer_util.predict_sequence(
            frame_predictions=frame_predictions,
            onset_predictions=onset_predictions,
            offset_predictions=offset_predictions,
            active_onsets=active_onsets,
            velocity_values=None,
            hparams=self.hparams,
            min_pitch=constants.MIN_MIDI_PITCH,
            instrument=1
        )
        sequence.notes.extend(infer_util.predict_sequence(
            frame_predictions=y_true[0][0],
            onset_predictions=y_true[1][0],
            offset_predictions=y_true[2][0],
            velocity_values=None,
            hparams=self.hparams,
            min_pitch=constants.MIN_MIDI_PITCH,
            instrument=0
        ).notes)
        midi_filename = f'{self.save_dir}/{epoch}_predicted.midi'
        midi_io.sequence_proto_to_midi_file(sequence, midi_filename)

    def save_stack_midi(self, y_probs, y_true, epoch):
        frame_predictions = y_true[0][0]
        onset_predictions = y_true[1][0]
        offset_predictions = y_true[2][0]
        sequence = infer_util.predict_sequence(
            frame_predictions=frame_predictions,
            onset_predictions=onset_predictions,
            offset_predictions=offset_predictions,
            velocity_values=None,
            hparams=self.hparams,
            min_pitch=constants.MIN_MIDI_PITCH,
            instrument=0)

        for i in range(3):
            # Output midi values for each stack (frames, onsets, offsets).
            sequence.notes.extend(infer_util.predict_sequence(
                frame_predictions=y_probs[i][0] > 0.5,
                onset_predictions=None,
                offset_predictions=None,
                velocity_values=None,
                hparams=self.hparams,
                min_pitch=constants.MIN_MIDI_PITCH,
                instrument=i + 1).notes)

        midi_filename = f'{self.save_dir}/{epoch}_stacks.midi'
        midi_io.sequence_proto_to_midi_file(sequence, midi_filename)

    def on_train_batch_begin(self, *args):
        pass

    def on_train_batch_end(self, *args):
        pass

    @abstractmethod
    def predict(self, X, y, epoch=None):
        pass

    def on_epoch_end(self, epoch, logs=None, model=None):
        if model:
            self.model = model
        x, y = self.generator.get()
        try:
            metrics = self.predict(x, y, epoch=epoch)
        except Exception as e:
            print(e)
            return
        if type(metrics) is dict:
            metrics_dict = metrics
        else:
            metrics_dict = metrics._asdict()
        self.metrics_history.append(metrics_dict)
        try:
            for name, value in metrics_dict.items():
                print('{} metrics:'.format(name))
                print('Precision: {}, Recall: {}, F1: {}\n'.format(value['precision'].numpy() * 100,
                                                                   value['recall'].numpy() * 100,
                                                                   value['f1_score'].numpy() * 100))
        except:
            print('Precision: {}, Recall: {}, F1: {}\n'.format(
                metrics_dict['note_precision'].numpy() * 100,
                metrics_dict['note_recall'].numpy() * 100,
                metrics_dict['note_f1_score'].numpy() * 100))
        gc.collect()


class MidiPredictionMetrics(MetricsCallback):
    def load_metrics(self, metrics_history):
        self.metrics_history = [MidiPredictionOutputMetrics(*x) for x in metrics_history]

    def predict(self, X, y, epoch=None):
        y_probs = self.model.predict_on_batch(X)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_midi(y_probs, y, epoch)
        self.save_stack_midi(y_probs, y, epoch)
        if self.note_based:
            sequence_label = infer_util.predict_sequence(y[0][0], None, None, None,
                                                         constants.MIN_MIDI_PITCH, self.hparams)
            return calculate_metrics(
                frame_predictions=y_probs[0][0] > self.hparams.predict_frame_threshold,
                onset_predictions=y_probs[1][0] > self.hparams.predict_onset_threshold,
                offset_predictions=y_probs[2][0] > self.hparams.predict_offset_threshold,
                velocity_values=y_probs[0][0] > self.hparams.predict_frame_threshold,  # ignored
                sequence_label=tf.convert_to_tensor(sequence_label.SerializeToString()),
                frame_labels=y[0][0],
                sequence_id=K.constant('', tf.string),
                hparams=self.hparams,
                min_pitch=constants.MIN_MIDI_PITCH,
                max_pitch=constants.MAX_MIDI_PITCH)
        frame_metrics = calculate_frame_metrics(y[0],
                                                y_probs[0] > self.hparams.predict_frame_threshold)
        onset_metrics = calculate_frame_metrics(y[1],
                                                y_probs[1] > self.hparams.predict_onset_threshold)
        offset_metrics = calculate_frame_metrics(y[2],
                                                 y_probs[2] > self.hparams.predict_offset_threshold)

        return MidiPredictionOutputMetrics(frame_metrics, onset_metrics, offset_metrics)


class TimbrePredictionMetrics(MetricsCallback):

    def load_metrics(self, metrics_history):
        self.metrics_history = [TimbrePredictionOutputMetrics(*x) for x in metrics_history]

    def predict(self, X, y, epoch=None):
        y_probs = self.model.predict_on_batch(X)
        print(y_probs + K.cast_to_floatx(y[0]))
        scores = flatten_f1_wrapper()(y[0], y_probs)
        del y_probs
        return TimbrePredictionOutputMetrics(scores)


class FullPredictionMetrics(MetricsCallback):
    def load_metrics(self, metrics_history):
        self.metrics_history = [MidiPredictionOutputMetrics(*x) for x in metrics_history]

    def predict(self, X, y, epoch=None):
        y_probs = self.model.predict_on_batch(X)
        frame_metrics = multi_track_prf_wrapper(self.hparams.predict_frame_threshold,
                                                self.hparams.multiple_instruments_threshold,
                                                only_f1=False, hparams=self.hparams)(y[0],
                                                                                     y_probs[0])
        onset_metrics = multi_track_prf_wrapper(self.hparams.predict_onset_threshold,
                                                self.hparams.multiple_instruments_threshold,
                                                only_f1=False, hparams=self.hparams)(y[1],
                                                                                     y_probs[1])
        offset_metrics = multi_track_prf_wrapper(self.hparams.predict_offset_threshold,
                                                 self.hparams.multiple_instruments_threshold,
                                                 only_f1=False, hparams=self.hparams)(y[2],
                                                                                      y_probs[2])
        # Save agnostic midi.
        self.save_midi([get_last_channel(p) for p in y_probs], [get_last_channel(t) for t in y],
                       epoch)
        self.save_stack_midi([get_last_channel(p) for p in y_probs],
                             [get_last_channel(t) for t in y], epoch)

        del y_probs
        return MidiPredictionOutputMetrics(frame_metrics, onset_metrics, offset_metrics)


class EvaluationMetrics(MetricsCallback):
    def __init__(self, is_full=True, **kwargs):
        self.is_full = is_full
        if self.is_full:
            self.num_classes = 12 + 1
        else:
            self.num_classes = 12
        super().__init__(**kwargs)

    def predict(self, X, y, epoch=None):
        y_probs = self.model.call(X, training=False)

        if self.is_full:
            agnostic_predictions = get_last_channel(
                y_probs[0]) > self.hparams.predict_frame_threshold
            multi_stack_predictions = convert_multi_instrument_probs_to_predictions(
                y_probs[0],
                self.hparams.predict_frame_threshold,
                self.hparams.multiple_instruments_threshold)[0]
            permuted_stack_predictions = K.permute_dimensions(multi_stack_predictions,
                                                              (tf.rank(multi_stack_predictions) - 1,
                                                               *K.arange(tf.rank(
                                                                   multi_stack_predictions) - 1)))
            permuted_predictions = K.concatenate(
                [permuted_stack_predictions, agnostic_predictions], axis=0)
        else:
            timbre_probs = y_probs[0]
            top_probs = K.cast(tf.one_hot(
                K.argmax(timbre_probs),
                K.int_shape(timbre_probs)[-1]), 'bool')
            frame_predictions = tf.logical_or(
                timbre_probs > self.hparams.multiple_instruments_threshold,
                tf.logical_and(top_probs,
                               timbre_probs > 0.5))
            permuted_predictions = K.permute_dimensions(frame_predictions,
                                                        (tf.rank(frame_predictions) - 1,
                                                         *K.arange(tf.rank(
                                                             frame_predictions) - 1)))
        y_relevant = y[0][0]
        permuted_true = K.permute_dimensions(y_relevant, (
            tf.rank(y_relevant) - 1, *K.arange(tf.rank(y_relevant) - 1)))

        instrument_metrics = dict()
        for i in range(self.num_classes):
            instrument_metric = calculate_frame_metrics(permuted_true[i],
                                                        permuted_predictions[i])
            instrument_metrics[constants.FAMILY_IDX_STRINGS[i]] = instrument_metric

        if self.is_full:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            # Save agnostic midi.
            self.save_midi([get_last_channel(p) for p in y_probs], [get_last_channel(t) for t in y],
                           epoch)
            self.save_stack_midi([get_last_channel(p) for p in y_probs],
                                 [get_last_channel(t) for t in y], epoch)

        del y_probs
        return instrument_metrics
