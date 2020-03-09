import functools

import tensorflow as tf
from dotmap import DotMap

from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription.accuracy_util import flatten_accuracy_wrapper, \
    flatten_loss_wrapper
from magenta.models.onsets_frames_transcription.layer_util import conv_bn_elu_layer, \
    get_all_croppings, time_distributed_wrapper
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import he_normal, VarianceScaling
from tensorflow.keras.layers import BatchNormalization, Conv2D, \
    Dense, Dropout, \
    Input, concatenate, Lambda, Reshape, LSTM, \
    Flatten, ELU, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

class FullModel:
    def __init__(self, hparams):
        self.hparams = hparams