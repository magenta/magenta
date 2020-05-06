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

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.metrics import categorical_accuracy
from sklearn.metrics import precision_recall_fscore_support, classification_report

from magenta.models.polyamp.metrics import f1_score, accuracy_without_true_negatives


def _convert_to_multi_instrument_predictions(y_true, y_probs,
                                             threshold=0.5,
                                             multiple_instruments_threshold=0.6,
                                             hparams=None):
    flat_y_predictions = tf.reshape(y_probs, (-1, K.int_shape(y_probs)[-1]))
    y_predictions = convert_multi_instrument_probs_to_predictions(flat_y_predictions,
                                                                  threshold,
                                                                  multiple_instruments_threshold,
                                                                  hparams)
    flat_y_true = _remove_last_channel(tf.reshape(K.cast(y_true, 'bool'),
                                                  (-1, K.int_shape(y_true)[-1])))

    return flat_y_true, y_predictions


def _remove_last_channel(y_probs):
    permuted_probs = K.permute_dimensions(y_probs,
                                          (tf.rank(y_probs) - 1,
                                           *K.arange(tf.rank(y_probs) - 1)))
    timbre_probs = K.permute_dimensions(permuted_probs[:-1],
                                        (*K.arange(tf.rank(permuted_probs) - 1) + 1, 0))
    return timbre_probs


def get_last_channel(y):
    """
    Permute and get the last channel, which is the instrument-agnostic
    information.
    :param y: Output values to get the last channel from.
    :return: The instrument-agnostic probability.
    """
    permuted_y = (
        K.permute_dimensions(y, (tf.rank(y) - 1, *K.arange(tf.rank(y) - 1)))
    )
    return permuted_y[-1]


def convert_multi_instrument_probs_to_predictions(y_probs,
                                                  threshold=0.5,
                                                  multiple_instruments_threshold=0.6,
                                                  hparams=None):
    """
    Convert probabilities into boolean predictions for
    each instrument.
    :param y_probs: Predicted outputs.
    :param threshold: Threshold for instrument-agnostic predictions.
    :param multiple_instruments_threshold: Threshold for predicting
    more than one instrument.
    :param hparams: Hyperparameters
    :return: The boolean multi-instrument predictions.
    """
    if hparams is None or hparams.timbre_final_activation == 'sigmoid':
        # Use last dimension as instrument-agnostic probability.
        agnostic_predictions = K.expand_dims(get_last_channel(y_probs) > threshold)
        timbre_probs = _remove_last_channel(y_probs)
        top_probs = K.cast(tf.one_hot(
            K.argmax(timbre_probs),
            K.int_shape(timbre_probs)[-1]), 'bool')
        timbre_predictions = tf.logical_or(timbre_probs > multiple_instruments_threshold,
                                           tf.logical_and(top_probs,
                                                          timbre_probs > 0.5))
        return tf.logical_and(agnostic_predictions, timbre_predictions)

    sum_probs = K.sum(y_probs, axis=-1)
    # Remove any where the original melodic prediction
    # is below the threshold.
    thresholded_y_probs = y_probs * K.expand_dims(K.cast_to_floatx(sum_probs > threshold))
    # Get the label for the highest probability instrument.
    one_hot = tf.one_hot(tf.reshape(tf.nn.top_k(y_probs).indices, K.int_shape(y_probs)[:-1]),
                         K.int_shape(y_probs)[-1])
    # Only predict the best instrument at each location.
    times_one_hot = thresholded_y_probs * one_hot

    present_mean = 1 / K.sum(K.cast_to_floatx(thresholded_y_probs > 0), -1)
    y_predictions = tf.logical_or(thresholded_y_probs > K.expand_dims(present_mean),
                                  times_one_hot > 0)
    return y_predictions


def single_track_present_accuracy_wrapper(threshold=0.5):
    """
    Get the instrument-agnostic accuracy for the Full Model.
    :param threshold: Threshold for Melodic probabilities.
    :param hparams: Hyperparameters.
    :return: Single-track binary accuracy.
    """

    def single_present_acc_fn(y_true, y_probs):
        reshaped_y_true = K.reshape(y_true, (-1, y_true.shape[-1]))
        reshaped_y_probs = K.reshape(y_probs, (-1, y_probs.shape[-1]))

        single_y_true = get_last_channel(reshaped_y_true)
        single_y_predictions = get_last_channel(reshaped_y_probs) > threshold
        frame_metrics = calculate_frame_metrics(single_y_true, single_y_predictions)
        print('Precision: {}, Recall: {}, F1: {}'.format(frame_metrics['precision'].numpy() * 100,
                                                         frame_metrics['recall'].numpy() * 100,
                                                         frame_metrics['f1_score'].numpy() * 100))
        return calculate_frame_metrics(single_y_true, single_y_predictions)[
            'accuracy_without_true_negatives']

    return single_present_acc_fn


def multi_track_present_accuracy_wrapper(threshold=0.5,
                                         multiple_instruments_threshold=0.6,
                                         hparams=None):
    """
    Get the multi-track accuracy for the Full Model.
    :param threshold: Threshold for Melodic probabilities.
    :param multiple_instruments_threshold: Threshold for Timbre
    probabilities.
    :param hparams: Hyperparameters.
    :return: Multi-track binary accuracy.
    """

    def present_acc_fn(y_true, y_probs):
        flat_y_true, flat_y_predictions = (
            _convert_to_multi_instrument_predictions(y_true,
                                                     y_probs,
                                                     threshold,
                                                     multiple_instruments_threshold,
                                                     hparams=hparams)
        )
        acc = tf.logical_and(tf.equal(K.sum(K.cast_to_floatx(flat_y_predictions), -1),
                                      K.sum(K.cast_to_floatx(flat_y_true), -1)),
                             tf.equal(K.argmax(K.cast_to_floatx(flat_y_predictions), -1),
                                      K.argmax(K.cast_to_floatx(flat_y_true), -1)))
        print(f'acc: {K.mean(K.cast_to_floatx(acc))}')

        frame_true_positives = tf.reduce_sum(K.cast_to_floatx(tf.logical_and(
            flat_y_true,
            tf.equal(flat_y_true, flat_y_predictions))))

        return (frame_true_positives
                / (frame_true_positives
                   + tf.reduce_sum(K.cast_to_floatx(
                            tf.not_equal(flat_y_true, flat_y_predictions)
                        ))))

    return present_acc_fn


def multi_track_prf_wrapper(threshold=0.5, multiple_instruments_threshold=0.6,
                            print_report=False, only_f1=True, hparams=None):
    """
    Get the Full Model PRF scores.
    :param threshold: Threshold for Melodic probabilities.
    :param multiple_instruments_threshold: Threshold for Timbre
    probabilities.
    :param print_report: Whether to print to console.
    :param only_f1: Include only F1 or include P, R, and F1.
    :param hparams: Hyperparameters.
    :return: Either the PRF or the F1-Score.
    """

    def multi_track_prf_fn(y_true, y_probs):
        flat_y_true, flat_y_predictions = (
            _convert_to_multi_instrument_predictions(y_true,
                                                     y_probs,
                                                     threshold,
                                                     multiple_instruments_threshold,
                                                     hparams)
        )

        flat_y_predictions = K.cast_to_floatx(flat_y_predictions)
        flat_y_true = K.cast_to_floatx(flat_y_true)
        ignoring_melodic = (flat_y_predictions
                            * K.expand_dims(
                    K.flatten(get_last_channel(y_true))))
        individual_sums = K.sum(K.cast(ignoring_melodic, 'int32'), 0)

        print(f'num_agnostic: '
              f"""{K.sum(K.cast_to_floatx(get_last_channel(y_probs) > threshold
                                          / hparams.prediction_generosity))}""")
        print(f'true_num_agnostic: {K.sum(K.cast_to_floatx(get_last_channel(y_true) > 0))}')
        print(f'both: '
              f"""{K.sum(K.cast_to_floatx(get_last_channel(y_probs) > threshold
                                          / hparams.prediction_generosity)
                         * K.cast_to_floatx(get_last_channel(y_true) > 0))}""")
        print(f'total predicted {K.sum(individual_sums)}')
        if print_report:
            print(classification_report(flat_y_true, ignoring_melodic,
                                        digits=4, zero_division=0))
            print([f'{i}:{x}' for i, x in enumerate(individual_sums)])

        # Definitely don't use macro accuracy here
        # because some instruments won't be present.
        precision, recall, f1, _ = (
            precision_recall_fscore_support(flat_y_true,
                                            ignoring_melodic,
                                            average='weighted',
                                            zero_division=0)
        )
        scores = {
            'precision': K.constant(precision),
            'recall': K.constant(recall),
            'f1_score': K.constant(f1)
        }
        return scores

    def _f1(t, p):
        return multi_track_prf_fn(t, p)['f1_score']

    if only_f1:
        return _f1
    return multi_track_prf_fn


def binary_accuracy_wrapper(threshold=0.5):
    """
    Get the Melodic binary accuracy while removing any true negatives.
    :param threshold: Threshold for Melodic probabilities.
    :return: The accuracy without true negatives.
    """

    def binary_accuracy_fn(labels, probs):
        return calculate_frame_metrics(labels, probs > threshold)[
            'accuracy_without_true_negatives'
        ]

    return binary_accuracy_fn


def f1_wrapper(threshold=0.5):
    """
    Get Melodic F1-Score scores after flattening the inputs.
    :param threshold: Threshold for Melodic probabilities.
    :return: F1-Score
    """

    def _f1(labels, probs):
        return calculate_frame_metrics(labels, probs > threshold)[
            'f1_score'
        ]

    return _f1


def flatten_f1_wrapper(threshold=0.5):
    """
    Get Timbre PRF scores after flattening the inputs.
    :param threshold: Threshold for Timbre probabilities.
    :return: Precision, Recall, and F1-Score
    """

    def flatten_f1_fn(y_true, y_probs):
        print([f'{i}:{x}' for i, x in
               enumerate(K.max(K.reshape(y_probs, (-1, K.int_shape(y_probs)[-1])), 0))])
        y_predictions = K.cast_to_floatx(y_probs > threshold)
        y_predictions = K.reshape(y_predictions, (-1, K.int_shape(y_predictions)[-1]))

        reshaped_y_true = K.reshape(y_true, (-1, K.int_shape(y_true)[-1]))

        # Remove any predictions from padded values.
        y_predictions = y_predictions * K.expand_dims(
            K.cast_to_floatx(K.sum(reshaped_y_true, -1) > 0))

        print(classification_report(reshaped_y_true, y_predictions,
                                    digits=4, zero_division=0))
        print([f'{i}:{x}' for i, x in enumerate(K.cast(K.sum(y_predictions, 0), 'int32'))])
        precision, recall, f1, _ = (
            precision_recall_fscore_support(reshaped_y_true,
                                            y_predictions,
                                            average='micro',
                                            zero_division=0)
        )
        scores = {
            'precision': K.constant(precision),
            'recall': K.constant(recall),
            'f1_score': K.constant(f1)
        }
        return scores

    return flatten_f1_fn


def flatten_accuracy_wrapper():
    """Return categorical accuracy after flattening
    the truths and predictions.
    """

    def flatten_accuracy_fn(y_true, y_pred):
        rebatched_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
        rebatched_true = K.reshape(y_true, (-1, K.int_shape(y_true)[-1]))

        # Remove any predictions from padded values.
        rebatched_pred = rebatched_pred * K.expand_dims(
            K.cast_to_floatx(K.sum(rebatched_true, -1) > 0))
        return categorical_accuracy(rebatched_true, rebatched_pred)

    return flatten_accuracy_fn


def calculate_frame_metrics(frame_labels, frame_predictions):
    # Copyright 2020 The Magenta Authors.
    # Modifications by Jack Spencer Smith
    """Calculate frame-based metrics."""
    frame_labels_bool = tf.cast(frame_labels, tf.bool)
    frame_predictions_bool = tf.cast(frame_predictions, tf.bool)

    frame_true_positives = tf.reduce_sum(K.cast_to_floatx(tf.logical_and(
        K.equal(frame_labels_bool, True),
        K.equal(frame_predictions_bool, True))))
    frame_false_positives = tf.reduce_sum(K.cast_to_floatx(tf.logical_and(
        K.equal(frame_labels_bool, False),
        K.equal(frame_predictions_bool, True))))
    frame_false_negatives = tf.reduce_sum(K.cast_to_floatx(tf.logical_and(
        K.equal(frame_labels_bool, True),
        K.equal(frame_predictions_bool, False))))
    frame_accuracy = (
            tf.reduce_sum(
                K.cast_to_floatx(tf.equal(frame_labels_bool, frame_predictions_bool))) /
            K.cast_to_floatx(tf.size(frame_labels))
    )

    frame_precision = tf.where(frame_true_positives + frame_false_positives > 0,
                               frame_true_positives
                               / (frame_true_positives + frame_false_positives),
                               0)
    frame_recall = tf.where(frame_true_positives + frame_false_negatives > 0,
                            frame_true_positives
                            / (frame_true_positives + frame_false_negatives),
                            0)
    frame_f1_score = f1_score(frame_precision, frame_recall)
    frame_accuracy_without_true_negatives = accuracy_without_true_negatives(
        frame_true_positives, frame_false_positives, frame_false_negatives)

    return {
        'true_positives': frame_true_positives,
        'false_positives': frame_false_positives,
        'false_negatives': frame_false_negatives,
        'accuracy': frame_accuracy,
        'accuracy_without_true_negatives': frame_accuracy_without_true_negatives,
        'precision': frame_precision,
        'recall': frame_recall,
        'f1_score': frame_f1_score,
    }
