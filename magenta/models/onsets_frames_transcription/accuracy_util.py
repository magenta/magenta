from collections import namedtuple
import tensorflow as tf

# 'name' should be a string
# 'method' should be a string or a function
from magenta.models.onsets_frames_transcription.metrics import calculate_frame_metrics
from tensorflow.keras.metrics import binary_accuracy

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

def f1_wrapper(threshold):
    def f1(labels, probs):
        # return binary_accuracy(labels, probs, threshold)

        return calculate_frame_metrics(labels, probs > threshold)['f1_score']


    return f1
