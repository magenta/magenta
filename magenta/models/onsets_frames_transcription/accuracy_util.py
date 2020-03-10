from collections import namedtuple
import tensorflow as tf
import tensorflow.keras.backend as K

# 'name' should be a string
# 'method' should be a string or a function
from tensorflow.keras.losses import categorical_crossentropy
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
                                                                   average='weighted')  # TODO maybe 'macro'
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