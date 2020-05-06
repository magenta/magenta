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


def _recall_weighing_loss(labels, predictions, epsilon=1e-9, recall_weighing=0):
    """Log Loss with Recall Weighing

    :param labels: The ground truth values.
    :param predictions: The predicted values.
    :param epsilon: Add this to prevent log of zero.
    :param recall_weighing: This is a scalar to weigh
    the trues by this many times more than falses.

    :return: Individual log losses.
    """

    predictions = K.cast_to_floatx(predictions)
    labels = K.cast_to_floatx(labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    losses = -labels * K.log(predictions + epsilon) - (1 - labels) * K.log(
        1 - predictions + epsilon)
    if recall_weighing != 0:
        if recall_weighing < 0:
            # Weigh towards precision if negative.
            labels = 1 - labels
            recall_weighing = -recall_weighing
            losses = losses * (labels + 1 / recall_weighing) / (1 + 1 / recall_weighing)

        else:
            losses = losses * (labels + 1 / recall_weighing) / (1 + 1 / recall_weighing)

    return losses


def _get_instrument_loss(permuted_y_true, instrument_probs,
                         instrument_idx, hparams,
                         recall_weighing=0, epsilon=1e-9):
    instrument_loss = tf.reduce_mean(
        _recall_weighing_loss(permuted_y_true[instrument_idx],
                              instrument_probs + epsilon,
                              epsilon=epsilon,
                              recall_weighing=recall_weighing
                                              * 1.5
                                              * hparams.family_recall_weight[
                                                  instrument_idx]))
    if K.sum(permuted_y_true[instrument_idx]) == 0:
        # Still learn from samples without that instrument
        # present; don't mess up under-represented instruments.
        instrument_loss *= 1e-2
    return instrument_loss


def melodic_loss_wrapper(recall_weighing=0):
    """
    Compute the loss for the Melodic Model.
    :param recall_weighing: Scalar to prefer recall over precision.
    :return: Melodic Model loss.
    """

    def melodic_loss_fn(label_true, label_predicted):
        return tf.reduce_mean(_recall_weighing_loss(label_true,
                                                    label_predicted,
                                                    recall_weighing=recall_weighing))

    return melodic_loss_fn


def full_model_loss_wrapper(hparams, recall_weighing=0):
    """
    Compute the loss for the Full Model.
    :param hparams: Hyperparameters.
    :param recall_weighing: Scalar to prefer recall over precision.
    :return: Full Model Loss.
    """

    def full_model_loss_fn(y_true, y_probs):
        # Permute to: num_instruments, batch, time, pitch.
        permuted_y_true = K.permute_dimensions(y_true, (3, 0, 1, 2))
        permuted_y_probs = K.permute_dimensions(y_probs, (3, 0, 1, 2))

        loss_list = []

        # Add Timbre Model loss.
        for instrument_idx in range(hparams.timbre_num_classes):
            ignore_melodic_probs = permuted_y_probs[instrument_idx] \
                                   * K.cast_to_floatx(permuted_y_true[-1])

            loss_list.append(_get_instrument_loss(permuted_y_true,
                                                  ignore_melodic_probs,
                                                  instrument_idx,
                                                  hparams=hparams,
                                                  recall_weighing=recall_weighing))

        instrument_debug_info = (
            [f'{i}: {x:.4f}. {K.max(permuted_y_probs[i]):.3f}'
             for i, x in enumerate(loss_list)]
        )
        print(f'total loss: {tf.reduce_sum(loss_list)} '
              f'{instrument_debug_info}')

        # Add Melodic Model loss.
        return tf.reduce_mean(loss_list) + K.mean(
            _recall_weighing_loss(permuted_y_true[-1],
                                  permuted_y_probs[-1],
                                  recall_weighing=recall_weighing))

    return full_model_loss_fn


def timbre_loss_wrapper(hparams,
                        recall_weighing=4):
    """
    Compute the loss for the Timbre Model.
    :param hparams: Scalar to prefer recall over precision.
    :param recall_weighing: Scalar to prefer recall over precision.
    :return: Timbre Model loss.
    """

    def timbre_loss_fn(y_true, y_probs):
        # Permute to: num_instruments, batch.
        permuted_y_true = K.transpose(
            K.reshape(K.cast_to_floatx(y_true), (-1, y_true.shape[-1])))
        permuted_y_probs = K.transpose(
            K.reshape(y_probs, (-1, y_probs.shape[-1])))
        permuted_y_probs = (permuted_y_probs
                            * K.expand_dims(K.cast_to_floatx(K.sum(permuted_y_true, 0) > 0), 0))
        loss_list = []

        for instrument_idx in range(hparams.timbre_num_classes):
            loss_list.append(
                _get_instrument_loss(permuted_y_true,
                                     permuted_y_probs[instrument_idx],
                                     instrument_idx,
                                     hparams=hparams,
                                     recall_weighing=recall_weighing)
            )

        return tf.reduce_mean(loss_list)

    return timbre_loss_fn
