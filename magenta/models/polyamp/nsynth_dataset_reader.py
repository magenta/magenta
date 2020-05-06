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

import functools

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow.keras.backend as K

from magenta.models.polyamp import dataset_reader, timbre_dataset_reader
from magenta.music import audio_io


def _parse_nsynth_example(example_proto):
    features = {
        'pitch': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        'instrument': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        'instrument_family': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        'velocity': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        'audio': tf.io.VarLenFeature(dtype=tf.float32),
    }
    record = tf.io.parse_single_example(example_proto, features)
    return record


def _get_approx_note_length(samples):
    window = 2048
    pooled_samples = np.max(
        np.abs([np.max(samples.numpy()[i:i + window]) for i in
                range(0, len(samples.numpy()), window)]),
        axis=-1)
    argmax = np.argmax(pooled_samples)
    max_amplitude = np.max(pooled_samples)
    low_amplitudes = np.ndarray.flatten(np.argwhere(pooled_samples < 0.1 * max_amplitude))
    try:
        first_low = np.ndarray.flatten(np.argwhere(low_amplitudes > argmax))[0]
    except KeyError:
        return len(samples)
    else:
        release_idx = low_amplitudes[first_low]
    length = len(samples) * release_idx / len(pooled_samples)
    return int(round(length))


# combine the batched datasets' audio together
def reduce_audio_in_batch(tensor, hparams=None, is_training=True):
    instrument_count = hparams.timbre_training_max_instruments
    note_croppping_list = []
    instrument_family_list = []
    samples_list = []
    max_length = 0
    for i in range(instrument_count):
        pitch = tensor['pitch'][i]
        # Move the audio so there are different attack times.
        start_idx = tf.random.uniform((), minval=0, maxval=hparams.timbre_max_start_offset,
                                      dtype='int64')
        samples = K.concatenate([
            tf.zeros(start_idx),
            tf.sparse.to_dense(tensor['audio'])[i]
        ])

        end_idx = (start_idx
                   + tf.py_function(_get_approx_note_length,
                                    [tf.sparse.to_dense(tensor['audio'])[i]],
                                    tf.int64))
        if hparams.timbre_max_len and end_idx > hparams.timbre_max_len:
            samples = tf.slice(samples, begin=[0], size=[hparams.timbre_max_len])
            end_idx = hparams.timbre_max_len
        if len(samples) > max_length:
            max_length = len(samples)

        samples_list.append(samples)

        instrument_family = tensor['instrument_family'][i]
        note_croppping_list.append(timbre_dataset_reader.NoteCropping(
            pitch=pitch,
            start_idx=start_idx,
            end_idx=end_idx
        ))
        instrument_family_list.append(tf.one_hot(tf.cast(instrument_family, tf.int32),
                                                 hparams.timbre_num_classes))

    # Pad the end of the shorter audio clips.
    samples_list = list(map(
        lambda x: tf.pad(x, [[0, max_length - len(x)]]), samples_list
    ))

    combined_samples = (tf.reduce_sum(tf.convert_to_tensor(samples_list), axis=0)
                        / instrument_count)

    # Ensure all audios in batches are the same length.
    if hparams.timbre_max_len:
        pad_length = hparams.timbre_max_len
    else:
        pad_length = hparams.timbre_max_start_offset + 5 * hparams.sample_rate
    combined_samples = tf.pad(combined_samples,
                              [[0, pad_length - tf.shape(combined_samples)[0]]])
    note_croppings = tf.convert_to_tensor(note_croppping_list, dtype=tf.int32)
    instrument_families = tf.convert_to_tensor(instrument_family_list, dtype=tf.int32)

    wav_data = tf.py_function(
        lambda x: audio_io.samples_to_wav_data(x.numpy(), sample_rate=hparams.sample_rate),
        [combined_samples],
        tf.string
    )

    return dict(
        audio=wav_data,
        note_croppings=note_croppings,
        instrument_families=instrument_families,
    )


def provide_batch(examples,
                  hparams,
                  is_training,
                  shuffle_examples,
                  skip_n_initial_records,
                  for_full_model=False,
                  **kwargs):
    """
    Returns batches of tensors read from TFRecord files.
    Reads from NSynth-like datasets.
    :param examples: TFRecord filenames.
    :param hparams: Hyperparameters.
    :param is_training: Is this a training dataset.
    :param shuffle_examples: Randomly shuffle the examples.
    :param skip_n_initial_records: Skip n records in the dataset.
    :param for_full_model: Keep melodic information or discard it.
    :param kwargs: Unused.
    :return: TensorFlow batched dataset.
    """
    hparams = params

    input_dataset = dataset_reader.read_examples(
        examples, is_training, shuffle_examples, skip_n_initial_records, hparams)

    input_dataset = input_dataset.map(_parse_nsynth_example)
    reduced_dataset = input_dataset.batch(hparams.timbre_training_max_instruments).map(
        functools.partial(reduce_audio_in_batch,
                          hparams=hparams,
                          is_training=is_training)
    )
    if for_full_model:
        dataset = timbre_dataset_reader.process_for_full_model(reduced_dataset,
                                                               is_training,
                                                               hparams)
    else:
        spec_dataset = reduced_dataset.map(
            functools.partial(timbre_dataset_reader.include_spectrogram, hparams=hparams)
        )
        model_input = spec_dataset.map(functools.partial(
            timbre_dataset_reader.timbre_input_tensors_to_model_input,
            hparams=hparams, is_training=is_training
        ))
        dataset = model_input.batch(hparams.nsynth_batch_size)

    return dataset.prefetch(buffer_size=1)
