from collections import namedtuple
import tensorflow as tf

# 'name' should be a string
# 'method' should be a string or a function
from magenta.models.onsets_frames_transcription.metrics import calculate_frame_metrics

AccuracyMetric = namedtuple('AccuracyMetric', ('name', 'method'))


def Dixon(yTrue, yPred):
    from keras import backend as K

    # true (correct) positives, predicted positives = tp + fp, real (ground-truth) positives = tp + fn
    tp, pp, rp = K.sum(yTrue * K.round(yPred)), K.sum(K.round(yPred)), K.sum(yTrue)
    return 1 if pp == 0 and rp == 0 else tp / (pp + rp - tp + K.epsilon())

def boolean_accuracy_wrapper(threshold):
    def acc(labels, probs):
        predictions_bool = probs > threshold
        labels_bool = tf.cast(labels, tf.bool)
        return (tf.reduce_sum(
                    tf.cast(tf.equal(labels_bool, predictions_bool), tf.float32)) /
                tf.cast(tf.size(labels), tf.float32))
    return acc
