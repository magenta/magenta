from collections import namedtuple
from itertools import product

import tensorflow as tf
import tensorflow.keras.backend as K

# 'name' should be a string
# 'method' should be a string or a function
from tensorflow.keras.losses import categorical_crossentropy, CategoricalCrossentropy, Reduction
from tensorflow.keras.metrics import categorical_accuracy
from sklearn.metrics import precision_recall_fscore_support, classification_report

from magenta.common import tf_utils
from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription.metrics import calculate_frame_metrics

AccuracyMetric = namedtuple('AccuracyMetric', ('name', 'method'))


def Dixon(yTrue, yPred):
    from keras import backend as K

    # true (correct) positives, predicted positives = tp + fp, real (ground-truth) positives = tp + fn
    tp, pp, rp = K.sum(yTrue * K.round(yPred)), K.sum(K.round(yPred)), K.sum(yTrue)
    return 1 if pp == 0 and rp == 0 else tp / (pp + rp - tp + K.epsilon())


def multi_track_loss_wrapper(recall_weighing=0, epsilon=1e-9):
    def multi_track_loss(y_true, y_probs):
        # num_instruments, batch, time, pitch
        permuted_y_true = K.permute_dimensions(y_true, (3, 0, 1, 2))
        permuted_y_probs = K.permute_dimensions(y_probs, (3, 0, 1, 2))

        loss_list = []
        for instrument_idx in range(K.int_shape(permuted_y_true)[0]):
            if 0 == K.sum(permuted_y_probs[instrument_idx]) == K.sum(
                    permuted_y_true[instrument_idx]):
                # ignore the non-present instruments
                continue
            loss_list.append(K.mean(tf_utils.log_loss(permuted_y_true[instrument_idx],
                                                      permuted_y_probs[instrument_idx] + epsilon,
                                                      epsilon=epsilon,
                                                      recall_weighing=recall_weighing)))
        # add instrument-independent loss
        loss_list.append(K.mean(tf_utils.log_loss(K.max(y_true, axis=-1),
                                                  K.sum(y_probs, axis=-1) + epsilon,
                                                  epsilon=epsilon,
                                                  recall_weighing=recall_weighing)))
        return tf.reduce_mean(loss_list)

    return multi_track_loss


def convert_to_multi_instrument_predictions(y_true, y_probs, threshold=0.5,
                                            multiple_instruments_threshold=0.2):
    y_predictions = convert_multi_instrument_probs_to_predictions(y_probs,
                                                                  threshold,
                                                                  multiple_instruments_threshold)
    flat_y_predictions = tf.reshape(y_predictions, (-1, K.int_shape(y_predictions)[-1]))
    flat_y_true = tf.reshape(K.cast(y_true, 'bool'), (-1, K.int_shape(y_true)[-1]))
    return flat_y_true, flat_y_predictions


def convert_multi_instrument_probs_to_predictions(y_probs,
                                                  threshold,
                                                  multiple_instruments_threshold=0.2):
    sum_probs = K.sum(y_probs, axis=-1)
    # remove any where the original midi prediction is below the threshold
    thresholded_y_probs = y_probs * K.expand_dims(K.cast_to_floatx(sum_probs > threshold))
    # get the label for the highest probability instrument
    one_hot = tf.one_hot(tf.reshape(tf.nn.top_k(y_probs).indices, K.int_shape(y_probs)[:-1]),
                         K.int_shape(y_probs)[-1])
    # only predict the best instrument at each location
    times_one_hot = thresholded_y_probs * one_hot
    y_predictions = tf.logical_or(thresholded_y_probs > multiple_instruments_threshold,
                                  times_one_hot > 0)
    return y_predictions


def single_track_present_accuracy_wrapper(threshold):
    def single_present_acc(y_true, y_probs):
        single_y_true = K.max(y_true, axis=-1)
        single_y_predictions = K.sum(y_probs, axis=-1) > threshold
        return calculate_frame_metrics(single_y_true, single_y_predictions)[
            'accuracy_without_true_negatives']

    return single_present_acc

def multi_track_present_accuracy_wrapper(threshold, multiple_instruments_threshold=0.2):
    def present_acc(y_true, y_probs):
        flat_y_true, flat_y_predictions = convert_to_multi_instrument_predictions(y_true,
                                                                                  y_probs,
                                                                                  threshold,
                                                                                  multiple_instruments_threshold)
        return calculate_frame_metrics(flat_y_true, flat_y_predictions)[
            'accuracy_without_true_negatives']

    return present_acc


def multi_track_prf_wrapper(threshold, multiple_instruments_threshold=0.2, print_report=False, only_f1=True):
    def multi_track_prf(y_true, y_probs):
        flat_y_true, flat_y_predictions = convert_to_multi_instrument_predictions(y_true,
                                                                                  y_probs,
                                                                                  threshold,
                                                                                  multiple_instruments_threshold)

        print(f'total predicted {K.sum(K.cast_to_floatx(flat_y_predictions))}')
        if print_report:
            print(classification_report(flat_y_true, flat_y_predictions, zero_division=1))

        # definitely don't use macro accuracy here because some instruments won't be present
        precision, recall, f1, _ = precision_recall_fscore_support(flat_y_true,
                                                                   flat_y_predictions,
                                                                   average='micro',
                                                                   zero_division=1)  # TODO maybe 0
        scores = {
            'precision': K.constant(precision),
            'recall': K.constant(recall),
            'f1_score': K.constant(f1)
        }
        return scores

    def f1(t, p):
        return multi_track_prf(t, p)['f1_score']

    if only_f1:
        return f1
    return multi_track_prf


def binary_accuracy_wrapper(threshold):
    def acc(labels, probs):
        # return binary_accuracy(labels, probs, threshold)

        return calculate_frame_metrics(labels, probs > threshold)['accuracy_without_true_negatives']

    return acc


def true_positive_wrapper(threshold):
    def pos(labels, probs):
        # return binary_accuracy(labels, probs, threshold)

        return calculate_frame_metrics(labels, probs > threshold)['true_positives']

    return pos


def f1_wrapper(threshold):
    def f1(labels, probs):
        # return binary_accuracy(labels, probs, threshold)

        return calculate_frame_metrics(labels, probs > threshold)['f1_score']

    return f1


def flatten_f1_wrapper(hparams):
    def flatten_f1_fn(y_true, y_probs):
        y_predictions = tf.one_hot(K.flatten(tf.nn.top_k(y_probs).indices), y_probs.shape[-1],
                                   dtype=tf.int32)

        reshaped_y_true = K.reshape(y_true, (-1, y_predictions.shape[-1]))

        print(classification_report(reshaped_y_true, y_predictions))

        precision, recall, f1, _ = precision_recall_fscore_support(reshaped_y_true,
                                                                   y_predictions,
                                                                   average='weighted')  # TODO maybe 'macro'
        scores = {
            'precision': K.constant(precision),
            'recall': K.constant(recall),
            'f1_score': K.constant(f1)
        }
        return scores

    return flatten_f1_fn


# use epsilon to prevent nans when doing log
def flatten_loss_wrapper(hparams, epsilon=1e-9):
    def flatten_loss_fn(y_true, y_pred):
        if hparams.timbre_coagulate_mini_batches:
            return categorical_crossentropy(y_true, y_pred + epsilon,
                                            label_smoothing=hparams.timbre_label_smoothing)
        rebatched_pred = K.reshape(y_pred, (-1, y_pred.shape[-1]))
        # using y_pred on purpose because keras thinks y_true shape is (None, None, None)
        rebatched_true = K.reshape(y_true, (-1, y_pred.shape[-1]))
        return categorical_crossentropy(rebatched_true, rebatched_pred + epsilon,
                                        label_smoothing=hparams.timbre_label_smoothing)

    return flatten_loss_fn


def flatten_accuracy_wrapper(hparams):
    def flatten_accuracy_fn(y_true, y_pred):
        if hparams.timbre_coagulate_mini_batches:
            # remove any predictions from padded values
            y_pred = y_pred * K.cast_to_floatx(K.sum(y_true, -1) > 0)
            return categorical_accuracy(y_true, y_pred)
        rebatched_pred = K.reshape(y_pred, (-1, y_pred.shape[-1]))
        # using y_pred on purpose because keras thinks y_true shape is (None, None, None)
        rebatched_true = K.reshape(y_true, (-1, y_pred.shape[-1]))

        # remove any predictions from padded values
        rebatched_pred = rebatched_pred * K.expand_dims(K.cast_to_floatx(K.sum(rebatched_true, -1) > 0))
        return categorical_accuracy(rebatched_true, rebatched_pred)

    return flatten_accuracy_fn


class WeightedCategoricalCrossentropy(CategoricalCrossentropy):

    def __init__(
            self,
            weights,
            from_logits=False,
            label_smoothing=0,
            reduction=Reduction.SUM_OVER_BATCH_SIZE,
            name='categorical_crossentropy',
    ):
        super().__init__(
            from_logits, label_smoothing, reduction, name=f"weighted_{name}"
        )
        self.weights = weights

    def call(self, y_true_unshaped, y_pred_unshaped):
        y_pred = K.reshape(y_pred_unshaped, (-1, y_pred_unshaped.shape[-1]))
        # using y_pred on purpose because keras thinks y_true shape is (None, None, None)
        y_true = K.reshape(K.cast(y_true_unshaped, K.floatx()), (-1, y_pred_unshaped.shape[-1]))

        # remove any predictions from padded values
        y_pred = y_pred * K.expand_dims(K.cast_to_floatx(K.sum(y_true, -1) > 0)) + 1e-9

        weights = self.weights
        nb_cl = len(weights)
        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=1)
        y_pred_max = K.reshape(
            y_pred_max, (K.shape(y_pred)[0], 1))
        y_pred_max_mat = K.cast(
            K.equal(y_pred, y_pred_max), K.floatx())
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += (
                    weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        return super().call(y_true, y_pred) * final_mask
