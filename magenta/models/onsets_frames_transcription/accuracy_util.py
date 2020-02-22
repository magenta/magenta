from collections import namedtuple


# 'name' should be a string
# 'method' should be a string or a function
from magenta.models.onsets_frames_transcription.metrics import calculate_frame_metrics

AccuracyMetric = namedtuple('AccuracyMetric', ('name', 'method'))

def Dixon(yTrue, yPred):
    from keras import backend as K

    # true (correct) positives, predicted positives = tp + fp, real (ground-truth) positives = tp + fn
    tp, pp, rp = K.sum(yTrue * K.round(yPred)), K.sum(K.round(yPred)), K.sum(yTrue)
    return 1 if pp == 0 and rp == 0 else tp / (pp + rp - tp + K.epsilon())

def frames_accuracy(labels, predictions):
    calculate_frame_metrics(
        frame_labels=labels,
        frame_predictions=predictions)