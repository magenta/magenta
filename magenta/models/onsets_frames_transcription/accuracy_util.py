from collections import namedtuple
from itertools import product

import tensorflow as tf
import tensorflow.keras.backend as K

# 'name' should be a string
# 'method' should be a string or a function
from tensorflow.keras import losses
from keras.layers import Multiply
from tensorflow.keras.losses import categorical_crossentropy, CategoricalCrossentropy, Reduction
from tensorflow.keras.metrics import categorical_accuracy
from sklearn.metrics import precision_recall_fscore_support, classification_report
from tensorflow_core.python.keras.engine.training_utils import get_loss_function
from tensorflow_core.python.keras.losses import LossFunctionWrapper

from magenta.common import tf_utils
from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription.metrics import calculate_frame_metrics, \
    accuracy_without_true_negatives

AccuracyMetric = namedtuple('AccuracyMetric', ('name', 'method'))


def Dixon(yTrue, yPred):
    from keras import backend as K

    # true (correct) positives, predicted positives = tp + fp, real (ground-truth) positives = tp + fn
    tp, pp, rp = K.sum(yTrue * K.round(yPred)), K.sum(K.round(yPred)), K.sum(yTrue)
    return 1 if pp == 0 and rp == 0 else tp / (pp + rp - tp + K.epsilon())


def multi_track_loss_wrapper(hparams, recall_weighing=0, epsilon=1e-9):
    def weighted_loss(y_true_unshaped, y_pred_unshaped):
        y_pred = K.reshape(y_pred_unshaped, (-1, y_pred_unshaped.shape[-1]))
        # using y_pred on purpose because keras thinks y_true shape is (None, None, None)
        y_true = K.reshape(K.cast(y_true_unshaped, K.floatx()), (-1, y_pred_unshaped.shape[-1]))
        weights = K.ones((K.int_shape(y_true)[-1], K.int_shape(y_true)[-1])) * (1 + tf.linalg.diag(
            tf.repeat(
                hparams.weight_correct_multiplier - 1,
                hparams.timbre_num_classes
            )
        ))
        support_inv_exp = 2.
        total_support = tf.math.pow(K.expand_dims(K.sum(y_true, 0) + 10., 0),
                                    1 / (1 + support_inv_exp))
        # weigh those with less support more
        weighted_weights = total_support * weights / (
            tf.math.pow(K.expand_dims(K.sum(y_true, 0) + 10., -1),
                        1 / support_inv_exp))

        nb_cl = len(weighted_weights)
        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=1)
        y_pred_max = K.reshape(
            y_pred_max, (K.shape(y_pred)[0], 1))
        y_pred_max_mat = K.cast(
            K.equal(y_pred, y_pred_max), K.floatx())
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += (
                    weighted_weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        return categorical_crossentropy(y_true, y_pred + epsilon) * final_mask

    def multi_track_loss(y_true, y_probs):
        # num_instruments, batch, time, pitch
        permuted_y_true = K.permute_dimensions(y_true, (3, 0, 1, 2))
        permuted_y_probs = K.permute_dimensions(y_probs, (3, 0, 1, 2))

        loss_list = []
        total_mean = K.mean(K.max(y_true, -1))

        for instrument_idx in range(hparams.timbre_num_classes):
            # 50/50 for something not in melody prediction
            ignore_melodic_probs = permuted_y_probs[instrument_idx] \
                                   * K.cast_to_floatx(permuted_y_true[-1])
            instrument_loss = tf.reduce_mean(
                tf_utils.log_loss(permuted_y_true[instrument_idx],
                                  ignore_melodic_probs + epsilon,
                                  epsilon=epsilon,
                                  recall_weighing=recall_weighing
                                                  * 1.5
                                                  * hparams.family_recall_weight[instrument_idx]))

            # instrument_loss *= hparams.inv_frequency_weight[instrument_idx]
            if K.sum(permuted_y_true[instrument_idx]) == 0:
                # Still learn a little from samples without the instrument present
                # Don't mess up under-represented instruments much
                instrument_loss *= 1e-2

            loss_list.append(instrument_loss)

        print(
            f'total loss: {tf.reduce_sum(loss_list)} {[f"{i}: {x:.4f}. {K.max(permuted_y_probs[i]):.3f}" for i, x in enumerate(loss_list)]}')

        # add instrument-independent loss
        return tf.reduce_mean(loss_list) + K.mean(
            tf_utils.log_loss(permuted_y_true[-1],
                              permuted_y_probs[-1] + epsilon,
                              epsilon=epsilon,
                              recall_weighing=recall_weighing))
        # return K.sum(loss_list)

    # return lambda *x: K.mean(weighted_loss(*x))
    return lambda *x: multi_track_loss(*x)  # + K.mean(weighted_loss(*x))


def convert_to_multi_instrument_predictions(y_true, y_probs, threshold=0.5,
                                            multiple_instruments_threshold=0.2, hparams=None):
    flat_y_predictions = tf.reshape(y_probs, (-1, K.int_shape(y_probs)[-1]))
    y_predictions = convert_multi_instrument_probs_to_predictions(flat_y_predictions,
                                                                  threshold,
                                                                  multiple_instruments_threshold,
                                                                  hparams)
    flat_y_true = remove_last_channel(tf.reshape(K.cast(y_true, 'bool'),
                                                 (-1, K.int_shape(y_true)[-1])))

    return flat_y_true, y_predictions


def remove_last_channel(y_probs):
    permuted_probs = K.permute_dimensions(y_probs,
                                          (tf.rank(y_probs) - 1,
                                           *K.arange(tf.rank(y_probs) - 1)))
    timbre_probs = K.permute_dimensions(permuted_probs[:-1],
                                        (*K.arange(tf.rank(permuted_probs) - 1) + 1, 0))
    return timbre_probs


def get_last_channel(y):
    permuted_y = K.permute_dimensions(y, (tf.rank(y) - 1, *K.arange(tf.rank(y) - 1)))
    return permuted_y[-1]


def convert_multi_instrument_probs_to_predictions(y_probs,
                                                  threshold,
                                                  multiple_instruments_threshold=0.5,
                                                  hparams=None):
    if hparams is None or hparams.timbre_final_activation == 'sigmoid':
        # use last dimension as instrument-agnostic probability
        agnostic_predictions = K.expand_dims(get_last_channel(y_probs) > threshold)
        # timbre_probs = remove_last_channel(y_probs)
        # return timbre_probs * K.expand_dims(agnostic_probs) > multiple_instruments_threshold
        timbre_probs = remove_last_channel(y_probs)
        top_probs = K.cast(tf.one_hot(
            K.argmax(timbre_probs),
            K.int_shape(timbre_probs)[-1]), 'bool')
        timbre_predictions = tf.logical_or(timbre_probs > multiple_instruments_threshold,
                                           tf.logical_and(top_probs,
                                                          timbre_probs > 0.5))
        return tf.logical_and(agnostic_predictions, timbre_predictions)

    sum_probs = K.sum(y_probs, axis=-1)
    # remove any where the original midi prediction is below the threshold
    thresholded_y_probs = y_probs * K.expand_dims(K.cast_to_floatx(sum_probs > threshold))
    # get the label for the highest probability instrument
    one_hot = tf.one_hot(tf.reshape(tf.nn.top_k(y_probs).indices, K.int_shape(y_probs)[:-1]),
                         K.int_shape(y_probs)[-1])
    # only predict the best instrument at each location
    times_one_hot = thresholded_y_probs * one_hot

    present_mean = 1 / K.sum(K.cast_to_floatx(thresholded_y_probs > 0), -1)
    y_predictions = tf.logical_or(thresholded_y_probs > K.expand_dims(present_mean),
                                  times_one_hot > 0)
    # y_predictions = thresholded_y_probs > multiple_instruments_threshold
    return y_predictions


def single_track_present_accuracy_wrapper(threshold):
    def single_present_acc(y_true, y_probs):
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

    return single_present_acc


def multi_track_present_accuracy_wrapper(threshold, multiple_instruments_threshold=0.2,
                                         hparams=None):
    def present_acc(y_true, y_probs):
        true_agnostic = K.expand_dims(get_last_channel(K.flatten(y_true)), -1) > 0
        predictions_agnostic = K.expand_dims(get_last_channel(K.flatten(y_probs))) > threshold
        flat_y_true, flat_y_predictions = convert_to_multi_instrument_predictions(y_true,
                                                                                  y_probs,
                                                                                  threshold,
                                                                                  multiple_instruments_threshold,
                                                                                  hparams=hparams)
        # flat_y_true = K.cast_to_floatx(flat_y_true)
        # flat_y_predictions = K.cast_to_floatx(flat_y_predictions)
        acc = tf.logical_and(tf.equal(K.sum(K.cast_to_floatx(flat_y_predictions), -1),
                                      K.sum(K.cast_to_floatx(flat_y_true), -1)),
                             tf.equal(K.argmax(K.cast_to_floatx(flat_y_predictions), -1),
                                      K.argmax(K.cast_to_floatx(flat_y_true), -1)))
        print(f'acc: {K.mean(K.cast_to_floatx(acc))}')

        frame_true_positives = tf.reduce_sum(K.cast_to_floatx(tf.logical_and(
            flat_y_true,
            tf.equal(flat_y_true, flat_y_predictions))))

        return frame_true_positives / (frame_true_positives + tf.reduce_sum(K.cast_to_floatx(
            tf.not_equal(flat_y_true, flat_y_predictions)
        )))

    return present_acc


def multi_track_prf_wrapper(threshold, multiple_instruments_threshold=0.5, print_report=False,
                            only_f1=True, hparams=None):
    def multi_track_prf(y_true, y_probs):
        flat_y_true, flat_y_predictions = convert_to_multi_instrument_predictions(y_true,
                                                                                  y_probs,
                                                                                  threshold,
                                                                                  multiple_instruments_threshold,
                                                                                  hparams)

        # remove any predictions from padded values <- Why would we do this? Ignore all empty????
        flat_y_predictions = K.cast_to_floatx(flat_y_predictions)
        flat_y_true = K.cast_to_floatx(flat_y_true)
        ignoring_melodic = flat_y_predictions * K.expand_dims(K.flatten(get_last_channel(y_true)))
        # flat_y_predictions = flat_y_predictions * K.expand_dims(
        #     K.cast_to_floatx(K.sum(K.cast_to_floatx(flat_y_true), -1) > 0))
        individual_sums = K.sum(K.cast(ignoring_melodic, 'int32'), 0)
        print(
            f'num_agnostic: {K.sum(K.cast_to_floatx(get_last_channel(y_probs) > threshold / hparams.prediction_generosity))}')
        print(f'true_num_agnostic: {K.sum(K.cast_to_floatx(get_last_channel(y_true) > 0))}')
        print(
            f'both: {K.sum(K.cast_to_floatx(get_last_channel(y_probs) > threshold / hparams.prediction_generosity) * K.cast_to_floatx(get_last_channel(y_true) > 0))}')
        print(f'total predicted {K.sum(individual_sums)}')
        if print_report:
            print(classification_report(flat_y_true, ignoring_melodic, digits=4, zero_division=0))
            # print(classification_report(K.max(K.cast_to_floatx(flat_y_true), axis=-1) > threshold,
            #                             K.sum(K.cast_to_floatx(flat_y_predictions), axis=-1) > 0,
            #                             digits=4,
            #                             zero_division=0))
            print([f'{i}:{x}' for i, x in enumerate(individual_sums)])

        # definitely don't use macro accuracy here because some instruments won't be present
        precision, recall, f1, _ = precision_recall_fscore_support(flat_y_true,
                                                                   ignoring_melodic,
                                                                   average='weighted',
                                                                   zero_division=0)  # TODO maybe 0
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


def flatten_f1_wrapper(hparams, threshold=0.5):
    def flatten_f1_fn(y_true, y_probs):
        # y_predictions = tf.one_hot(K.flatten(tf.nn.top_k(y_probs).indices), y_probs.shape[-1])
        y_predictions = K.cast_to_floatx(y_probs > threshold)
        y_predictions = K.reshape(y_predictions, (-1, K.int_shape(y_predictions)[-1]))

        reshaped_y_true = K.reshape(y_true, (-1, K.int_shape(y_true)[-1]))

        # remove any predictions from padded values
        y_predictions = y_predictions * K.expand_dims(
            K.cast_to_floatx(K.sum(reshaped_y_true, -1) > 0))

        print(classification_report(reshaped_y_true, y_predictions, digits=4, zero_division=0))
        print([f'{i}:{x}' for i, x in enumerate(K.cast(K.sum(y_predictions, 0), 'int32'))])
        precision, recall, f1, _ = precision_recall_fscore_support(reshaped_y_true,
                                                                   y_predictions,
                                                                   average='micro',
                                                                   zero_division=0)  # TODO maybe 'macro'
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
        rebatched_pred = rebatched_pred * K.expand_dims(
            K.cast_to_floatx(K.sum(rebatched_true, -1) > 0))
        return categorical_accuracy(rebatched_true, rebatched_pred)

    return flatten_accuracy_fn


def flatten_weighted_logit_loss(pos_weight=1):
    def weighted_logit_loss(y_true_unshaped, y_logits_unshaped):
        y_logits = K.reshape(y_logits_unshaped, (-1, y_logits_unshaped.shape[-1]))
        # using y_pred on purpose because keras thinks y_true shape is (None, None, None)
        y_true = K.reshape(K.cast(y_true_unshaped, K.floatx()), (-1, y_logits_unshaped.shape[-1]))

        # remove any predictions from padded values
        y_logits = y_logits + K.expand_dims(tf.where(K.sum(y_true, -1) > 0, 0.0, -1e+9))
        return tf.nn.weighted_cross_entropy_with_logits(y_true, y_logits, pos_weight=pos_weight)

    return weighted_logit_loss


class WeightedCrossentropy(LossFunctionWrapper):

    def __init__(
            self,
            weights,
            from_logits=False,
            label_smoothing=0,
            reduction=Reduction.SUM_OVER_BATCH_SIZE,
            name='categorical_crossentropy',
            recall_weighing=2,  # increase recall
    ):
        super().__init__(reduction, name=f"weighted_{name}")
        self.loss_fn = get_loss_function(name)
        self.from_logits = from_logits
        self.weights = weights
        self.recall_weighing = recall_weighing

    def call(self, y_true_unshaped, y_pred_unshaped):
        y_pred = K.reshape(y_pred_unshaped, (-1, y_pred_unshaped.shape[-1]))
        # using y_pred on purpose because keras thinks y_true shape is (None, None, None)
        # 0.25 for samples with that family somewhere within it
        y_true_unshaped = K.minimum(1, K.cast_to_floatx(y_true_unshaped) + 0.25 * K.cast_to_floatx(
            K.expand_dims(K.sum(y_true_unshaped, 1) > 0, 1)))
        y_true = K.reshape(K.cast(y_true_unshaped, K.floatx()), (-1, y_pred_unshaped.shape[-1]))

        # remove any predictions from padded values
        y_pred = y_pred * K.expand_dims(K.cast_to_floatx(K.sum(y_true, -1) > 0)) + 1e-9

        weights = self.weights
        support_inv_exp = 1.2
        total_support = tf.math.pow(K.expand_dims(K.sum(y_true, 0) + 10, 0),
                                    1 / (1.0 + support_inv_exp))
        # weigh those with less support more
        weighted_weights = total_support * weights / (
            tf.math.pow(K.expand_dims(K.sum(y_true, 0) + 10, -1), 1 / support_inv_exp))
        # print(weighted_weights)
        nb_cl = len(weights)
        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=1)
        y_pred_max = K.reshape(
            y_pred_max, (K.shape(y_pred)[0], 1))
        y_pred_max_mat = K.cast(
            K.equal(y_pred, y_pred_max), K.floatx())
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += (
                    weighted_weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])

        if self.from_logits:
            return K.sum(tf.nn.weighted_cross_entropy_with_logits(y_true,
                                                                  y_pred,
                                                                  pos_weight=self.recall_weighing),
                         -1) * final_mask
        if self.recall_weighing == 0:
            return self.loss_fn.call(y_true, y_pred) * final_mask
        return K.mean(tf_utils.log_loss(y_true, y_pred, recall_weighing=self.recall_weighing),
                      -1) * final_mask
