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

# Lint as: python3
"""Interface to asynchronously capture continuous audio from PyAudio.

This module provides one class, AudioRecorder, which buffers chunks of audio
from PyAudio.
"""

import importlib
import math
import queue
import time

from absl import logging
import numpy as np
import pyaudio
import scipy
import six

try:
  librosa = importlib.import_module('librosa')
  samplerate = None
except ModuleNotFoundError:
  try:
    samplerate = importlib.import_module('samplerate')
    librosa = None
  except ModuleNotFoundError:
    print('Either librosa or samplerate must be installed.')
    raise


def resample(audio, source_rate, target_rate):
  if librosa:
    return librosa.core.resample(
        audio, orig_sr=source_rate, target_sr=target_rate)
  if samplerate:
    ratio = float(target_rate) / source_rate
    return samplerate.resample(audio, ratio, 'sinc_best')


class AudioTimeoutError(Exception):
  """A timeout while waiting for pyaudio to buffer samples."""
  pass


class AudioRecorder(object):
  """Asynchronously record and buffer audio using pyaudio.

  This class wraps the pyaudio interface. It contains a queue.Queue object to
  hold chunks of raw audio, and a callback function _enqueue_audio() which
  places raw audio into this queue. This allows the pyaudio.Stream object to
  record asynchronously at low latency.

  The class acts as a context manager. When entering the context it creates a
  pyaudio.Stream object and starts recording; it stops recording on exit. The
  Stream saves all of its audio to the Queue as two-tuples of
  (timestamp, raw_audio). The raw_audio is available from the queue as a numpy
  array using the get_audio() function.

  This class uses the term "frame" in the same sense that PortAudio does, so
  "frame" means something different here than elsewhere in the daredevil stack.
  A frame in PortAudio is one audio sample across all channels, so one frame of
  16-bit stereo audio is four bytes of data as two 16-bit integers.
  """
  pyaudio_format = pyaudio.paFloat32
  numpy_format = np.float32
  num_channels = 1

  # How many frames of audio PyAudio will fetch at once.
  # Higher numbers will increase the latancy.
  frames_per_chunk = 2**9

  # Limit queue to this number of audio chunks.
  max_queue_chunks = 1200

  # Timeout if we can't get a chunk from the queue for timeout_factor times the
  # chunk duration.
  timeout_factor = 24

  def __init__(self,
               raw_audio_sample_rate_hz=48000,
               downsample_factor=3,
               device_index=None):
    self._downsample_factor = downsample_factor
    self._raw_audio_sample_rate_hz = raw_audio_sample_rate_hz
    self.audio_sample_rate_hz = (
        self._raw_audio_sample_rate_hz // self._downsample_factor)
    self._raw_audio_queue = queue.Queue(self.max_queue_chunks)
    self._audio = pyaudio.PyAudio()
    self._print_input_devices()
    self._device_index = device_index

  def __enter__(self):
    if self._device_index is None:
      self._device_index = self._audio.get_default_input_device_info()['index']
    kwargs = {'input_device_index': self._device_index}
    device_info = self._audio.get_device_info_by_host_api_device_index(
        0, self._device_index)
    if device_info.get('maxInputChannels') <= 0:
      raise ValueError('Audio device has insufficient input channels.')
    print("Using audio device '%s' for index %d" %
          (device_info['name'], device_info['index']))
    self._stream = self._audio.open(
        format=self.pyaudio_format,
        channels=self.num_channels,
        rate=self._raw_audio_sample_rate_hz,
        input=True,
        output=False,
        frames_per_buffer=self.frames_per_chunk,
        start=True,
        stream_callback=self._enqueue_raw_audio,
        **kwargs)
    logging.info('Started audio stream.')
    return self

  def __exit__(self, exception_type, exception_value, traceback):
    self._stream.stop_stream()
    self._stream.close()
    logging.info('Stopped and closed audio stream.')

  def __del__(self):
    self._audio.terminate()
    logging.info('Terminated PyAudio/PortAudio.')

  @property
  def is_active(self):
    return self._stream.is_active()

  @property
  def bytes_per_sample(self):
    return pyaudio.get_sample_size(self.pyaudio_format)

  @property
  def _chunk_duration_seconds(self):
    return self.frames_per_chunk / self._raw_audio_sample_rate_hz

  def _print_input_devices(self):
    info = self._audio.get_host_api_info_by_index(0)
    print('\nInput microphone devices:')
    for i in range(0, info.get('deviceCount')):
      device_info = self._audio.get_device_info_by_host_api_device_index(0, i)
      if device_info.get('maxInputChannels') <= 0:
        continue
      print('  ID: ', i, ' - ', device_info.get('name'))

  def _enqueue_raw_audio(self, in_data, *_):  # unused args to match expected
    try:
      self._raw_audio_queue.put((in_data, time.time()), block=False)
      return None, pyaudio.paContinue
    except queue.Full:
      error_message = 'Raw audio buffer full.'
      logging.error(error_message)
      raise AudioTimeoutError(error_message)

  def _get_chunk(self, timeout=None):
    raw_data, timestamp = self._raw_audio_queue.get(timeout=timeout)
    array_data = np.fromstring(raw_data, self.numpy_format).reshape(
        -1, self.num_channels)
    return array_data, timestamp

  def get_audio_device_info(self):
    if self._device_index is None:
      return self._audio.get_default_input_device_info()
    else:
      return self._audio.get_device_info_by_index(self._device_index)

  def sample_duration_seconds(self, num_samples):
    return num_samples / self.audio_sample_rate_hz / self.num_channels

  def clear_queue(self):
    logging.debug('Purging %d chunks from queue.',
                  self._raw_audio_queue.qsize())
    while not self._raw_audio_queue.empty():
      self._raw_audio_queue.get()

  def get_audio(self, num_audio_frames):
    """Grab at least num_audio_frames frames of audio.

    Record at least num_audio_frames of audio and transform it into a
    numpy array. The term "frame" is in the sense used by PortAudio; see the
    note in the class docstring for details.

    Audio returned will be the earliest audio in the queue; it could be from
    before this function was called.

    Args:
      num_audio_frames: minimum number of samples of audio to grab.

    Returns:
      A tuple of (audio, first_timestamp, last_timestamp).
    """
    num_audio_chunks = int(
        math.ceil(num_audio_frames * self._downsample_factor /
                  self.frames_per_chunk))
    logging.debug('Capturing %d chunks to get at least %d frames.',
                  num_audio_chunks, num_audio_frames)
    if num_audio_chunks < 1:
      num_audio_chunks = 1
    try:
      timeout = self.timeout_factor * self._chunk_duration_seconds
      chunks, timestamps = list(
          zip(*[
              self._get_chunk(timeout=timeout) for _ in range(num_audio_chunks)
          ]))
    except queue.Empty:
      error_message = 'Audio capture timed out after %.1f seconds.' % timeout
      logging.error(error_message)
      raise AudioTimeoutError(error_message)

    assert len(chunks) == num_audio_chunks
    logging.debug('Got %d chunks. Chunk 0 has shape %s and dtype %s.',
                  len(chunks), chunks[0].shape, chunks[0].dtype)
    if self._raw_audio_queue.qsize() > (0.8 * self.max_queue_chunks):
      logging.warning('%d chunks remain in the queue.',
                      self._raw_audio_queue.qsize())
    else:
      logging.debug('%d chunks remain in the queue.',
                    self._raw_audio_queue.qsize())

    audio = np.concatenate(chunks)
    if self._downsample_factor != 1:
      audio = resample(audio, self._raw_audio_sample_rate_hz,
                       self.audio_sample_rate_hz)

    logging.debug('Audio array has shape %s and dtype %s.', audio.shape,
                  audio.dtype)
    return audio, timestamps[0], timestamps[-1]


def wav_data_to_samples(wav_data, sample_rate):
  """Read PCM-formatted WAV data and return a NumPy array of samples.

  Uses scipy to read and librosa/samplerate to process WAV data. Audio will
  be converted to mono if necessary.

  Args:
    wav_data: WAV audio data to read.
    sample_rate: The number of samples per second at which the audio will be
      returned. Resampling will be performed if necessary.

  Returns:
    A numpy array of audio samples, single-channel (mono) and sampled at the
    specified rate, in float32 format.

  Raises:
    IOError: If scipy is unable to read the WAV data.
    IOError: If audio processing fails.
  """
  try:
    # Read the wav file, converting sample rate & number of channels.
    native_sr, y = scipy.io.wavfile.read(six.BytesIO(wav_data))
  except Exception as e:  # pylint: disable=broad-except
    raise IOError(e)

  if y.dtype == np.int16:  # Convert to float32.
    y = y.astype(np.float32) / np.iinfo(np.int16).max
  elif y.dtype == np.float32:  # Already float32.
    pass
  else:
    raise IOError('WAV file not 16-bit or 32-bit float PCM, unsupported')
  try:  # Convert to mono and the desired sample rate.
    if y.ndim == 2 and y.shape[1] == 2:
      y = np.mean(y, axis=1)
    if native_sr != sample_rate:
      y = resample(y, native_sr, sample_rate)
  except Exception as e:  # pylint: disable=broad-except
    raise IOError(e)
  return y
