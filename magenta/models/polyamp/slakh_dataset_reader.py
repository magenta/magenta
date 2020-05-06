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

import copy
import functools

import librosa
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow.keras.backend as K
from tensorflow_core import TensorShape

from magenta.models.polyamp import dataset_reader, instrument_family_mappings, \
    timbre_dataset_reader
from magenta.music import sequences_lib
from magenta.music.protobuf import music_pb2


def get_note_croppings(record, hparams=None, is_training=True):
    sequence = record['sequence']
    audio = record['audio']

    if is_training:
        audio = dataset_reader.transform_wav_data_op(
            audio,
            hparams=hparams,
            jitter_amount_sec=0)

    temp_hparams = copy.deepcopy(hparams)
    temp_hparams.spec_hop_length = hparams.timbre_hop_length
    temp_hparams.spec_type = hparams.timbre_spec_type
    temp_hparams.spec_log_amplitude = hparams.timbre_spec_log_amplitude
    spec = dataset_reader.wav_to_spec_op(audio, hparams=temp_hparams)
    if hparams.timbre_spec_log_amplitude:
        spec = spec - librosa.power_to_db(np.array([1e-9]))[0]
        spec /= K.max(spec)

    def get_note_croppings_fn(sequence_tensor):
        note_sequence = music_pb2.NoteSequence.FromString(sequence_tensor.numpy())
        note_sequence = sequences_lib.apply_sustain_control_changes(note_sequence)
        croppings = []
        families = []
        num_notes_ = 0
        for note in note_sequence.notes:
            note_family = instrument_family_mappings.midi_instrument_to_family[note.program]
            if note_family.value < hparams.timbre_num_classes:
                croppings.append(
                    timbre_dataset_reader.NoteCropping(
                        pitch=note.pitch,
                        start_idx=note.start_time * hparams.sample_rate,
                        end_idx=note.end_time * hparams.sample_rate)
                )
                families.append(tf.one_hot(tf.cast(
                    note_family.value,
                    tf.int32
                ), hparams.timbre_num_classes))
                num_notes_ += 1
        return croppings, families, num_notes_

    note_croppings, instrument_families, num_notes = tf.py_function(
        get_note_croppings_fn,
        [sequence],
        [tf.float32, tf.int32, tf.int32]
    )
    return dict(
        audio=audio,
        note_croppings=note_croppings,
        instrument_families=instrument_families,
        num_notes=num_notes
    )


def provide_batch(examples,
                  hparams,
                  is_training,
                  shuffle_examples,
                  skip_n_initial_records,
                  for_full_model=False,
                  **kwargs):
    """Returns batches of tensors read from TFRecord files.
    Reads from Slakh-like datasets.
    :param examples: TFRecord filenames.
    :param hparams: Hyperparameters.
    :param is_training: Is this a training dataset.
    :param shuffle_examples: Randomly shuffle the examples.
    :param skip_n_initial_records: Skip n records in the dataset.
    :param for_full_model: Keep melodic information or discard it.
    :param kwargs: Unused.
    :return: TensorFlow batched dataset.
    """

    input_dataset = dataset_reader.read_examples(
        examples, is_training, shuffle_examples, skip_n_initial_records, hparams)

    parse_example_fn = dataset_reader.parse_example

    input_dataset = input_dataset.map(parse_example_fn)
    reduced_dataset = input_dataset.map(functools.partial(get_note_croppings,
                                                          hparams=hparams,
                                                          is_training=is_training))
    reduced_dataset = reduced_dataset.filter(lambda x: x['num_notes'] > 0)

    if for_full_model:
        dataset = timbre_dataset_reader.process_for_full_model(reduced_dataset,
                                                               is_training,
                                                               hparams)

    else:
        spec_dataset = reduced_dataset.map(functools.partial(
            timbre_dataset_reader.include_spectrogram, hparams=hparams
        ))

        model_input = spec_dataset.map(functools.partial(
            timbre_dataset_reader.timbre_input_tensors_to_model_input,
            hparams=hparams, is_training=is_training
        ))

        dataset = model_input.padded_batch(
            hparams.slakh_batch_size,
            padded_shapes=(
                timbre_dataset_reader.FeatureTensors(
                    spec=TensorShape([None, 229, 1]),
                    note_croppings=TensorShape([None, 3]),
                ),
                timbre_dataset_reader.LabelTensors(
                    instrument_families=TensorShape([None, hparams.timbre_num_classes])
                )),
            padding_values=(
                timbre_dataset_reader.FeatureTensors(
                    spec=K.cast_to_floatx(0),
                    note_croppings=K.cast_to_floatx(-1e+7),  # negative value padding
                ),
                timbre_dataset_reader.LabelTensors(instrument_families=0)),
            drop_remainder=True)
    # Use `buffer_size=tf.data.experimental.AUTOTUNE` if possible.
    return dataset.prefetch(buffer_size=1)
