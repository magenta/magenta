# Copyright 2020 The Magenta Authors.
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

"""Data utility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.gansynth.lib import datasets
from magenta.models.gansynth.lib import train_util
from magenta.models.gansynth.lib.specgrams_helper import SpecgramsHelper
import tensorflow.compat.v1 as tf


class DataHelper(object):
  """A class for querying and converting data."""

  def __init__(self, config):
    self._config = config
    self._dataset_name = config['dataset_name']
    self.dataset = datasets.registry[self._dataset_name](config)
    self.specgrams_helper = self.make_specgrams_helper()

  def _map_fn(self):
    """Create a mapping function for the dataset."""
    raise NotImplementedError

  def make_specgrams_helper(self):
    """Create a specgrams helper for the dataset."""
    raise NotImplementedError

  def data_to_waves(self, data):
    """Converts data representation to waveforms."""
    raise NotImplementedError

  def waves_to_data(self, waves):
    """Converts data representation to waveforms."""
    raise NotImplementedError

  def get_pitch_counts(self):
    """Returns a dictionary {pitch value (int): count (int)}."""
    return self.dataset.get_pitch_counts()

  def provide_one_hot_labels(self, batch_size):
    """Returns a batch of one-hot labels."""
    with tf.name_scope('inputs'):
      with tf.device('/cpu:0'):
        return self.dataset.provide_one_hot_labels(batch_size=batch_size)

  def provide_data(self, batch_size):
    """Returns a batch of data and one-hot labels."""
    with tf.name_scope('inputs'):
      with tf.device('/cpu:0'):
        dataset = self.dataset.provide_dataset()
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.map(self._map_fn, num_parallel_calls=4)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)

        iterator = tf.data.make_initializable_iterator(dataset)
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                             iterator.initializer)

        data, one_hot_labels = iterator.get_next()
        data.set_shape([batch_size, None, None, None])
        one_hot_labels.set_shape([batch_size, None])
        return data, one_hot_labels


class DataSTFTHelper(DataHelper):
  """A data helper for Linear Spectrograms."""

  def make_specgrams_helper(self):
    final_resolutions = train_util.make_resolution_schedule(
        **self._config).final_resolutions
    return SpecgramsHelper(
        audio_length=self._config['audio_length'],
        spec_shape=final_resolutions,
        overlap=0.75,
        sample_rate=self._config['sample_rate'],
        mel_downscale=1,
        ifreq=True)

  def _map_fn(self, wave, one_hot_label):
    waves = wave[tf.newaxis, :, :]
    data = self.waves_to_data(waves)
    return data[0], one_hot_label

  def data_to_waves(self, data):
    return self.specgrams_helper.specgrams_to_waves(data)

  def waves_to_data(self, waves):
    return self.specgrams_helper.waves_to_specgrams(waves)


class DataWaveHelper(DataSTFTHelper):
  """A data helper for raw waveforms.

  For compatibility with the spectral network architectues, we add a second
  (redundant) channel and zero-pad along the time axis.
  """

  def make_specgrams_helper(self):
    return SpecgramsHelper(audio_length=64000,
                           spec_shape=(256, 512),
                           overlap=0.75,
                           sample_rate=self._config['sample_rate'],
                           mel_downscale=2)

  def data_to_waves(self, data):
    return data[:, 768:-768, 0, :1]

  def waves_to_data(self, waves):
    waves = waves[:, :, None, :]
    pad = tf.zeros([tf.shape(waves)[0], 768, 1, 1])
    waves = tf.concat([pad, waves, pad], axis=1)
    return tf.concat([waves, waves], axis=3)


class DataSTFTNoIFreqHelper(DataHelper):
  """A data helper for Linear Spectrograms."""

  def make_specgrams_helper(self):
    final_resolutions = train_util.make_resolution_schedule(
        **self._config).final_resolutions
    return SpecgramsHelper(
        audio_length=self._config['audio_length'],
        spec_shape=final_resolutions,
        overlap=0.75,
        sample_rate=self._config['sample_rate'],
        mel_downscale=1,
        ifreq=False)

  def _map_fn(self, wave, one_hot_label):
    waves = wave[tf.newaxis, :, :]
    data = self.waves_to_data(waves)
    return data[0], one_hot_label

  def data_to_waves(self, data):
    return self.specgrams_helper.specgrams_to_waves(data)

  def waves_to_data(self, waves):
    return self.specgrams_helper.waves_to_specgrams(waves)


class DataMelHelper(DataSTFTHelper):
  """A data helper for Mel Spectrograms."""

  def data_to_waves(self, data):
    return self.specgrams_helper.melspecgrams_to_waves(data)

  def waves_to_data(self, waves):
    return self.specgrams_helper.waves_to_melspecgrams(waves)


registry = {
    'linear': DataSTFTHelper,
    'phase': DataSTFTNoIFreqHelper,
    'mel': DataMelHelper,
    'wave': DataWaveHelper,
}
