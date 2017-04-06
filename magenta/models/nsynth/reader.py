# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module to load the Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import tensorflow as tf

# FFT Specgram Shapes
SPECGRAM_REGISTRY = {
    (nfft, hop): shape for nfft, hop, shape in zip(
        [256, 256, 512, 512, 1024, 1024],
        [64, 128, 128, 256, 256, 512],
        [[129, 1001, 2], [129, 501, 2], [257, 501, 2],
         [257, 251, 2], [513, 251, 2], [513, 126, 2]])
}


class NSynthDataset(object):
  """Dataset object to help manage the SSTable loading."""

  def __init__(self, tfrecord_path, is_training=True):
    self.is_training = is_training
    self.record_path = tfrecord_path

  def get_example(self, batch_size):
    reader = tf.TFRecordReader()
    num_epochs = None if self.is_training else 1
    capacity = batch_size
    path_queue = tf.train.input_producer(
        [self.record_path],
        num_epochs=num_epochs,
        shuffle=self.is_training,
        capacity=capacity)
    key, serialized_example = reader.read(path_queue)
    features = {
        "note_str": tf.FixedLenFeature([], dtype=tf.string),
        "pitch": tf.FixedLenFeature([1], dtype=tf.int64),
        "audio": tf.FixedLenFeature([64000], dtype=tf.float32),
        "qualities": tf.FixedLenFeature([10], dtype=tf.int64),
        "instrument_source": tf.FixedLenFeature([1], dtype=tf.int64),
        "instrument_family": tf.FixedLenFeature([1], dtype=tf.int64),
    }
    example = tf.parse_single_example(serialized_example, features)
    return example


  def get_wavenet_batch(self, batch_size, length=64000):
    """Get the Tensor expressions from the reader.

    Args:
      batch_size: The integer batch size.
      length: Number of timesteps of a cropped sample to produce.

    Returns:
      A dict of key:tensor pairs. This includes "pitch", "wav", and "key".
    """
    example = self.get_example(batch_size)
    wav = example["audio"]
    wav = tf.slice(wav, [0], [64000])
    pitch = tf.squeeze(example["pitch"])
    key = tf.squeeze(example["note_str"])

    if self.is_training:
      # random crop
      crop = tf.random_crop(wav, [length])
      crop = tf.reshape(crop, [1, length])
      key, crop, pitch = tf.train.shuffle_batch(
          [key, crop, pitch],
          batch_size,
          num_threads=4,
          capacity=500 * batch_size,
          min_after_dequeue=200 * batch_size)
    else:
      # fixed center crop
      offset = (64000 - length) // 2  # 24320
      crop = tf.slice(wav, [offset], [length])
      crop = tf.reshape(crop, [1, length])
      key, crop, pitch = tf.train.shuffle_batch(
          [key, crop, pitch],
          batch_size,
          num_threads=4,
          capacity=500 * batch_size,
          min_after_dequeue=200 * batch_size)

    crop = tf.reshape(tf.cast(crop, tf.float32), [batch_size, length])
    pitch = tf.cast(pitch, tf.int32)
    return {"pitch": pitch, "wav": crop, "key": key}


#   def get_baseline_batch(self, hparams):
#     """Get the Tensor expressions from the reader.

#     Args:
#       batch_size: The integer batch size.
#       length: Number of timesteps of a cropped sample to produce.

#     Returns:
#       A dict of key:tensor pairs. This includes "pitch", "wav", and "key".
#     """

#     example = self.get_example(batch_size)
#     wav = tf.slice(example["audio"], [0], [64000])
#     pitch = tf.squeeze(example["pitch"])
#     key = tf.squeeze(example["note_str"])
#     instrument_source = tf.squeeze(example["instrument_source"])
#     instrument_family = tf.squeeze(example["instrument_family"])
#     qualities = tf.slice(example["qualities"], [0], 10]

#     # program = tf.reshape(example.get("program", tf.constant(0)),
#     #                      [1, 1])  # only for earlier datasets
#     # pitch = tf.reshape(example.get("pitch", tf.constant(0)), [1, 1])
#     # velocity = tf.reshape(example.get("velocity", tf.constant(0)), [1, 1])
#     # instrument = tf.reshape(example.get("instrument", tf.constant(0)),
#     #                         [1, 1])  # uid for instrument
#     # instrument_family = tf.reshape(
#     #     example.get("instrument_family", tf.constant(0)),
#     #     [1, 1])  # uid for instrument
#     # instrument_source = tf.reshape(
#     #     example.get("instrument_source", tf.constant(0)),
#     #     [1, 1])  # uid for instrument
#     # qualities = tf.reshape(
#     #     example.get("qualities",
#     #                tf.constant(np.zeros(self.dataset.num_qualities))),
#     #     [1, self.dataset.num_qualities])  # uid for instrument
#     # audio = tf.slice(example["audio"], [0], [self.dataset.num_samples])
#     # audio = tf.reshape(audio, [1, self.dataset.num_samples])

#     # audio.set_shape([1, self.dataset.num_samples])
#     # key = tf.expand_dims(tf.expand_dims(key, 0), 0)
#     # key.set_shape([1, 1])

#     # Get Specgrams
#     hop_length = hparams.hop_length
#     n_fft = hparams.n_fft
#     if hop_length and n_fft:
#       specgram = utils.tf_specgram(
#           audio,
#           n_fft=n_fft,
#           hop_length=hop_length,
#           mask=hparams.mask,
#           log_mag=hparams.log_mag,
#           use_cqt=hparams.use_cqt,
#           re_im=hparams.re_im,
#           dphase=hparams.dphase,
#           mag_only=hparams.mag_only)
#       if hparams.use_cqt:
#         shape = [1] + [252, 1001, 2]
#       else:
#         shape = [1] + datasets.SPECGRAM_REGISTRY[(n_fft, hop_length)]
#         if hparams.mag_only:
#           shape[-1] = 1
#       specgram = tf.reshape(specgram, shape)
#       tf.logging.info("SPECGRAM BEFORE PADDING", specgram)

#       if hparams.pad:
#         # Pad and crop specgram to 256x256
#         num_padding = 2**int(np.ceil(np.log(shape[2]) / np.log(2))) - shape[2]
#         tf.logging.info("num_pading: %d" % num_padding)
#         specgram = tf.reshape(specgram, shape)
#         specgram = tf.pad(specgram, [[0, 0], [0, 0], [0, num_padding], [0, 0]])
#         specgram = tf.slice(specgram, [0, 0, 0, 0], [-1, shape[1] - 1, -1, -1])
#         tf.logging.info("SPECGRAM AFTER PADDING", specgram)

#     # Form a Batch
#     if self.is_training:
#       (key, audio, velocity, program, pitch, specgram, instrument,
#        instrument_source, instrument_family,
#        qualities) = tf.train.shuffle_batch(
#            [
#                key, audio, velocity, program, pitch, specgram, instrument,
#                instrument_source, instrument_family, qualities
#            ],
#            batch_size=hparams.batch_size,
#            capacity=20 * hparams.batch_size,
#            min_after_dequeue=10 * hparams.batch_size,
#            enqueue_many=True,
#            allow_smaller_final_batch=self.allow_smaller_final_batch)
#     elif hparams.batch_size > 1:
#       (key, audio, velocity, program, pitch, specgram, instrument,
#        instrument_source, instrument_family, qualities) = tf.train.batch(
#            [
#                key, audio, velocity, program, pitch, specgram, instrument,
#                instrument_source, instrument_family, qualities
#            ],
#            batch_size=hparams.batch_size,
#            capacity=10 * hparams.batch_size,
#            enqueue_many=True,
#            allow_smaller_final_batch=self.allow_smaller_final_batch)

#     audio.set_shape([hparams.batch_size, self.dataset.num_samples])

#     batch = BATCH(
#         key=key, program=program, pitch=pitch, velocity=velocity, audio=audio,
#         instrument=instrument, instrument_source=instrument_source,
#         instrument_family=instrument_family, qualities=qualities,
#         spectrogram=specgram)

#     return batch


# # All dataset names are in caps, pylint: disable=invalid-name
# # 4seconds * 16khz = 64000.
# NUM_SAMPLES_16k = 64000
# # 4seconds * 8khz = 32000.
# NUM_SAMPLES_8k = 32000

# NSYNTH_TRAIN = DATASET(
#     samples_per_second=16000,
#     path_train=os.path.join(DATA_DIR_PERM, "nsynth_rc4-train.tfrecord"),
#     path_test=os.path.join(DATA_DIR_PERM, "nsynth_rc4-valid.tfrecord"),
#     num_samples=NUM_SAMPLES_16k,
#     num_qualities=10,
#     features={
#         "velocity": tf.FixedLenFeature([1], tf.int64),
#         "pitch": tf.FixedLenFeature([1], tf.int64),
#         "sample_rate": tf.FixedLenFeature([1], tf.int64),
#         "audio": tf.FixedLenFeature([NUM_SAMPLES_16k], tf.float32),
#         "instrument": tf.FixedLenFeature([1], tf.int64),
#         "instrument_family": tf.FixedLenFeature([1], tf.int64),
#         "instrument_source": tf.FixedLenFeature([1], tf.int64),
#         "qualities": tf.FixedLenFeature([10], tf.int64),
#     })
