from collections import namedtuple
from itertools import product

import tensorflow as tf
import tensorflow.keras.backend as K

# 'name' should be a string
# 'method' should be a string or a function
from tensorflow.keras.losses import categorical_crossentropy, CategoricalCrossentropy, Reduction
from tensorflow.keras.metrics import categorical_accuracy
from sklearn.metrics import precision_recall_fscore_support, classification_report

from magenta.models.onsets_frames_transcription.metrics import calculate_frame_metrics

AccuracyMetric = namedtuple('AccuracyMetric', ('name', 'method'))


def Dixon(yTrue, yPred):
    from keras import backend as K

    # true (correct) positives, predicted positives = tp + fp, real (ground-truth) positives = tp + fn
    tp, pp, rp = K.sum(yTrue * K.round(yPred)), K.sum(K.round(yPred)), K.sum(yTrue)
    return 1 if pp == 0 and rp == 0 else tp / (pp + rp - tp + K.epsilon())


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
                                                                   average='macro')  # TODO maybe 'macro'
        scores = {
            'precision': K.constant(precision),
            'recall': K.constant(recall),
            'f1_score': K.constant(f1)
        }
        return scores
    return flatten_f1_fn

def flatten_loss_wrapper(hparams):
    def flatten_loss_fn(y_true, y_pred):
        if hparams.timbre_coagulate_mini_batches:
            return categorical_crossentropy(y_true, y_pred,
                                            label_smoothing=hparams.timbre_label_smoothing)
        rebatched_pred = K.reshape(y_pred, (-1, y_pred.shape[-1]))
        # using y_pred on purpose because keras thinks y_true shape is (None, None, None)
        rebatched_true = K.reshape(y_true, (-1, y_pred.shape[-1]))
        return categorical_crossentropy(rebatched_true, rebatched_pred,
                                        label_smoothing=hparams.timbre_label_smoothing)
    return flatten_loss_fn

def flatten_accuracy_wrapper(hparams):
    def flatten_accuracy_fn(y_true, y_pred):
        if hparams.timbre_coagulate_mini_batches:
            return categorical_accuracy(y_true, y_pred)
        rebatched_pred = K.reshape(y_pred, (-1, y_pred.shape[-1]))
        # using y_pred on purpose because keras thinks y_true shape is (None, None, None)
        rebatched_true = K.reshape(y_true, (-1, y_pred.shape[-1]))
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