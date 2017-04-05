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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

# internal imports
import tensorflow as tf
import numpy as np

from magenta.models.nsynth.baseline import datasets
from magenta.models.nsynth.baseline import utils

# A namedtuple representing the batch of data.
BATCH = namedtuple("batch", [
    "key", "audio", "program", "pitch", "velocity", "spectrogram",
    "instrument", "instrument_family", "instrument_source", "qualities"
])


class NSynthReader(object):
  """Responsible for ferrying data from SSTables into estimator-ready format."""

  def __init__(self,
               dataset,
               hparams,
               is_training,
               num_epochs=None,
               allow_smaller_final_batch=False):
    self.dataset = dataset
    self.hparams = hparams
    self.is_training = is_training
    self.num_epochs = num_epochs
    self.allow_smaller_final_batch = allow_smaller_final_batch

  def get_batch(self):
    """Get a batch of input and target data from the dataset.

    Returns:
      batch: A BATCH (see above) with keys for "audio", "velocity", "program",
        and "pitch".
    """
    tf.logging.info(self.dataset.path_train)
    if self.is_training:
      filenames = tf.gfile.Glob(self.dataset.path_train)
      tf.logging.info(filenames)
    else:
      filenames = tf.gfile.Glob(self.dataset.path_test)
    filename_queue = tf.train.string_input_producer(
        filenames,
        num_epochs=self.num_epochs,
        shuffle=self.is_training,
        name="filename_queue")

    if self.dataset.path_train.endswith(".tfrecord"):
      reader = tf.TFRecordReader()
    else:
      reader = tf.SSTableReader()

    key, serialized = reader.read(filename_queue)

    parsed = tf.parse_single_example(serialized, self.dataset.features)

    # We give these dummy variables if absent so that we can include them in the
    # batch processing below.
    program = tf.reshape(parsed.get("program", tf.constant(0)),
                         [1, 1])  # only for earlier datasets
    pitch = tf.reshape(parsed.get("pitch", tf.constant(0)), [1, 1])
    velocity = tf.reshape(parsed.get("velocity", tf.constant(0)), [1, 1])
    instrument = tf.reshape(parsed.get("instrument", tf.constant(0)),
                            [1, 1])  # uid for instrument
    instrument_family = tf.reshape(
        parsed.get("instrument_family", tf.constant(0)),
        [1, 1])  # uid for instrument
    instrument_source = tf.reshape(
        parsed.get("instrument_source", tf.constant(0)),
        [1, 1])  # uid for instrument
    qualities = tf.reshape(
        parsed.get("qualities",
                   tf.constant(np.zeros(self.dataset.num_qualities))),
        [1, self.dataset.num_qualities])  # uid for instrument
    audio = tf.slice(parsed["audio"], [0], [self.dataset.num_samples])
    audio = tf.reshape(audio, [1, self.dataset.num_samples])
    tf.summary.audio(
        "AudioFromQueue",
        audio,
        sample_rate=self.hparams.samples_per_second,
        max_outputs=1,)

    # Apply Mu-Law Encoding
    if self.hparams.mu_law_num != 0:
      tf.logging.info("Using mu-law compression of %.2f."
                      % self.hparams.mu_law_num)
      audio = tf.squeeze(audio)
      audio = utils.tf_mu_law(audio, mu=1.0 * self.hparams.mu_law_num)
      audio = tf.expand_dims(audio, 0)
    else:
      tf.logging.info("Not using mu-law compression.")
    audio.set_shape([1, self.dataset.num_samples])
    key = tf.expand_dims(tf.expand_dims(key, 0), 0)
    key.set_shape([1, 1])

    # Get Specgrams
    hop_length = self.hparams.hop_length
    n_fft = self.hparams.n_fft
    if hop_length and n_fft:
      specgram = utils.tf_specgram(
          audio,
          n_fft=n_fft,
          hop_length=hop_length,
          mask=self.hparams.mask,
          log_mag=self.hparams.log_mag,
          use_cqt=self.hparams.use_cqt,
          re_im=self.hparams.re_im,
          dphase=self.hparams.dphase,
          mag_only=self.hparams.mag_only)
      if self.hparams.use_cqt:
        shape = [1] + [252, 1001, 2]
      else:
        shape = [1] + datasets.SPECGRAM_REGISTRY[(n_fft, hop_length)]
        if self.hparams.mag_only:
          shape[-1] = 1
      specgram = tf.reshape(specgram, shape)
      tf.logging.info("SPECGRAM BEFORE PADDING", specgram)

      if self.hparams.pad:
        # Pad and crop specgram to 256x256
        num_padding = 2**int(np.ceil(np.log(shape[2]) / np.log(2))) - shape[2]
        tf.logging.info("num_pading: %d" % num_padding)
        specgram = tf.reshape(specgram, shape)
        specgram = tf.pad(specgram, [[0, 0], [0, 0], [0, num_padding], [0, 0]])
        specgram = tf.slice(specgram, [0, 0, 0, 0], [-1, shape[1] - 1, -1, -1])
        tf.logging.info("SPECGRAM AFTER PADDING", specgram)

    # Form a Batch
    if self.is_training:
      (key, audio, velocity, program, pitch, specgram, instrument,
       instrument_source, instrument_family,
       qualities) = tf.train.shuffle_batch(
           [
               key, audio, velocity, program, pitch, specgram, instrument,
               instrument_source, instrument_family, qualities
           ],
           batch_size=self.hparams.batch_size,
           capacity=20 * self.hparams.batch_size,
           min_after_dequeue=10 * self.hparams.batch_size,
           enqueue_many=True,
           allow_smaller_final_batch=self.allow_smaller_final_batch)
    elif self.hparams.batch_size > 1:
      (key, audio, velocity, program, pitch, specgram, instrument,
       instrument_source, instrument_family, qualities) = tf.train.batch(
           [
               key, audio, velocity, program, pitch, specgram, instrument,
               instrument_source, instrument_family, qualities
           ],
           batch_size=self.hparams.batch_size,
           capacity=10 * self.hparams.batch_size,
           enqueue_many=True,
           allow_smaller_final_batch=self.allow_smaller_final_batch)

    audio.set_shape([self.hparams.batch_size, self.dataset.num_samples])

    batch = BATCH(
        key=key, program=program, pitch=pitch, velocity=velocity, audio=audio,
        instrument=instrument, instrument_source=instrument_source,
        instrument_family=instrument_family, qualities=qualities,
        spectrogram=specgram)

    return batch

