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
"""Audio file helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import librosa
import numpy as np
import scipy
import six


class AudioIOException(BaseException):
  pass


class AudioIOReadException(AudioIOException):
  pass


class AudioIODataTypeException(AudioIOException):
  pass


def int16_samples_to_float32(y):
  """Convert int16 numpy array of audio samples to float32."""
  if y.dtype != np.int16:
    raise ValueError('input samples not int16')
  return y.astype(np.float32) / np.iinfo(np.int16).max


def float_samples_to_int16(y):
  """Convert floating-point numpy array of audio samples to int16."""
  if not issubclass(y.dtype.type, np.floating):
    raise ValueError('input samples not floating-point')
  return (y * np.iinfo(np.int16).max).astype(np.int16)


def wav_data_to_samples(wav_data, sample_rate):
  """Read PCM-formatted WAV data and return a NumPy array of samples.

  Uses scipy to read and librosa to process WAV data. Audio will be converted to
  mono if necessary.

  Args:
    wav_data: WAV audio data to read.
    sample_rate: The number of samples per second at which the audio will be
        returned. Resampling will be performed if necessary.

  Returns:
    A numpy array of audio samples, single-channel (mono) and sampled at the
    specified rate, in float32 format.

  Raises:
    AudioIOReadException: If scipy is unable to read the WAV data.
    AudioIOException: If audio processing fails.
  """
  try:
    # Read the wav file, converting sample rate & number of channels.
    native_sr, y = scipy.io.wavfile.read(six.BytesIO(wav_data))
  except Exception as e:  # pylint: disable=broad-except
    raise AudioIOReadException(e)

  if y.dtype == np.int16:
    # Convert to float32.
    y = int16_samples_to_float32(y)
  elif y.dtype == np.float32:
    # Already float32.
    pass
  else:
    raise AudioIOException(
        'WAV file not 16-bit or 32-bit float PCM, unsupported')
  try:
    # Convert to mono and the desired sample rate.
    if y.ndim == 2 and y.shape[1] == 2:
      y = y.T
      y = librosa.to_mono(y)
    if native_sr != sample_rate:
      y = librosa.resample(y, native_sr, sample_rate)
  except Exception as e:  # pylint: disable=broad-except
    raise AudioIOException(e)
  return y


def samples_to_wav_data(samples, sample_rate):
  """Converts floating point samples to wav data."""
  wav_io = six.BytesIO()
  scipy.io.wavfile.write(wav_io, sample_rate, float_samples_to_int16(samples))
  return wav_io.getvalue()


def crop_samples(samples, sample_rate, crop_beginning_seconds,
                 total_length_seconds):
  """Crop WAV data.

  Args:
    samples: Numpy Array containing samples.
    sample_rate: The sample rate at which to interpret the samples.
    crop_beginning_seconds: How many seconds to crop from the beginning of the
        audio.
    total_length_seconds: The desired duration of the audio. After cropping the
        beginning of the audio, any audio longer than this value will be
        deleted.

  Returns:
    A cropped version of the samples.
  """
  samples_to_crop = int(crop_beginning_seconds * sample_rate)
  total_samples = int(total_length_seconds * sample_rate)
  cropped_samples = samples[samples_to_crop:(samples_to_crop + total_samples)]
  return cropped_samples


def crop_wav_data(wav_data, sample_rate, crop_beginning_seconds,
                  total_length_seconds):
  """Crop WAV data.

  Args:
    wav_data: WAV audio data to crop.
    sample_rate: The sample rate at which to read the WAV data.
    crop_beginning_seconds: How many seconds to crop from the beginning of the
        audio.
    total_length_seconds: The desired duration of the audio. After cropping the
        beginning of the audio, any audio longer than this value will be
        deleted.

  Returns:
    A cropped version of the WAV audio.
  """
  y = wav_data_to_samples(wav_data, sample_rate=sample_rate)
  samples_to_crop = int(crop_beginning_seconds * sample_rate)
  total_samples = int(total_length_seconds * sample_rate)
  cropped_samples = y[samples_to_crop:(samples_to_crop + total_samples)]
  return samples_to_wav_data(cropped_samples, sample_rate)


def jitter_wav_data(wav_data, sample_rate, jitter_seconds):
  """Add silence to the beginning of the file.

  Args:
     wav_data: WAV audio data to prepend with silence.
     sample_rate: The sample rate at which to read the WAV data.
     jitter_seconds: Seconds of silence to prepend.

  Returns:
     A version of the WAV audio with jitter_seconds silence prepended.
  """

  y = wav_data_to_samples(wav_data, sample_rate=sample_rate)
  silence_samples = jitter_seconds * sample_rate
  new_y = np.concatenate((np.zeros(np.int(silence_samples)), y))
  return samples_to_wav_data(new_y, sample_rate)


def load_audio(audio_filename, sample_rate):
  """Loads an audio file.

  Args:
    audio_filename: File path to load.
    sample_rate: The number of samples per second at which the audio will be
        returned. Resampling will be performed if necessary.

  Returns:
    A numpy array of audio samples, single-channel (mono) and sampled at the
    specified rate, in float32 format.

  Raises:
    AudioIOReadException: If librosa is unable to load the audio data.
  """
  try:
    y, unused_sr = librosa.load(audio_filename, sr=sample_rate, mono=True)
  except Exception as e:  # pylint: disable=broad-except
    raise AudioIOReadException(e)
  return y


def make_stereo(left, right):
  """Combine two mono signals into one stereo signal.

  Both signals must have the same data type. The resulting track will be the
  length of the longer of the two signals.

  Args:
    left: Samples for the left channel.
    right: Samples for the right channel.

  Returns:
    The two channels combined into a stereo signal.

  Raises:
    AudioIODataTypeException: if the two signals have different data types.
  """
  if left.dtype != right.dtype:
    raise AudioIODataTypeException(
        'left channel is of type {}, but right channel is {}'.format(
            left.dtype, right.dtype))

  # Mask of valid places in each row
  lens = np.array([len(left), len(right)])
  mask = np.arange(lens.max()) < lens[:, None]

  # Setup output array and put elements from data into masked positions
  out = np.zeros(mask.shape, dtype=left.dtype)
  out[mask] = np.concatenate([left, right])
  return out.T


def normalize_wav_data(wav_data, sample_rate, norm=np.inf):
  """Normalizes wav data.

  Args:
     wav_data: WAV audio data to prepend with silence.
     sample_rate: The sample rate at which to read the WAV data.
     norm: See the norm argument of librosa.util.normalize.

  Returns:
     A version of the WAV audio that has been normalized.
  """

  y = wav_data_to_samples(wav_data, sample_rate=sample_rate)
  new_y = librosa.util.normalize(y, norm=norm)
  return samples_to_wav_data(new_y, sample_rate)
