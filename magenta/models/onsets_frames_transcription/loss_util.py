from magenta.common import flatten_maybe_padded_sequences, tf_utils
import tensorflow as tf

import tensorflow.keras.backend as K


def log_loss_wrapper(flat_probs):
    def log_loss_wrapper_fn(label_true, label_predicted):
        flat_labels = K.flatten(label_true)
        return tf.reduce_mean(tf_utils.log_loss(flat_labels, K.flatten(label_predicted)))

    return log_loss_wrapper_fn


def log_loss_flattener(labels, predictions):
    return tf.reduce_mean(tf_utils.log_loss(K.flatten(labels), K.flatten(predictions)))
