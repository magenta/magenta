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
"""Experimental realtime transcription demo."""

import multiprocessing
import threading

from absl import app
from absl import flags
import attr
from colorama import Fore
from colorama import Style
from magenta.models.onsets_frames_transcription.realtime import audio_recorder
from magenta.models.onsets_frames_transcription.realtime import tflite_model
import numpy as np

flags.DEFINE_string('model_path', 'onsets_frames_wavinput.tflite',
                    'File path of TFlite model.')
flags.DEFINE_string('mic', None, 'Optional: Input source microphone ID.')
flags.DEFINE_float('mic_amplify', 30.0, 'Multiply raw audio mic input')
flags.DEFINE_string(
    'wav_file', None,
    'If specified, will decode the first 10 seconds of this wav file.')
flags.DEFINE_integer(
    'sample_rate_hz', 16000,
    'Sample Rate. The model expects 16000. However, some microphones do not '
    'support sampling at this rate. In that case use --sample_rate_hz 48000 and'
    'the code will automatically downsample to 16000')
FLAGS = flags.FLAGS


class TfLiteWorker(multiprocessing.Process):
  """Process for executing TFLite inference."""

  def __init__(self, model_path, task_queue, result_queue):
    multiprocessing.Process.__init__(self)
    self._model_path = model_path
    self._task_queue = task_queue
    self._result_queue = result_queue
    self._model = None

  def setup(self):
    if self._model is not None:
      return

    self._model = tflite_model.Model(model_path=self._model_path)

  def run(self):
    self.setup()
    while True:
      task = self._task_queue.get()
      if task is None:
        self._task_queue.task_done()
        return
      task(self._model)
      self._task_queue.task_done()
      self._result_queue.put(task)


@attr.s
class AudioChunk(object):
  serial = attr.ib()
  samples = attr.ib(repr=lambda w: '{} {}'.format(w.shape, w.dtype))


class AudioQueue(object):
  """Audio queue."""

  def __init__(self, callback, audio_device_index, sample_rate_hz,
               model_sample_rate, frame_length, overlap):
    # Initialize recorder.
    downsample_factor = sample_rate_hz / model_sample_rate
    self._recorder = audio_recorder.AudioRecorder(
        sample_rate_hz,
        downsample_factor=downsample_factor,
        device_index=audio_device_index)

    self._frame_length = frame_length
    self._overlap = overlap

    self._audio_buffer = np.array([], dtype=np.int16).reshape(0, 1)
    self._chunk_counter = 0
    self._callback = callback

  def start(self):
    """Start processing the queue."""
    with self._recorder:
      timed_out = False
      while not timed_out:
        assert self._recorder.is_active
        new_audio = self._recorder.get_audio(self._frame_length -
                                             len(self._audio_buffer))
        audio_samples = np.concatenate(
            (self._audio_buffer, new_audio[0] * FLAGS.mic_amplify))

        # Extract overlapping
        first_unused_byte = 0
        for pos in range(0, audio_samples.shape[0] - self._frame_length,
                         self._frame_length - self._overlap):
          self._callback(
              AudioChunk(self._chunk_counter,
                         audio_samples[pos:pos + self._frame_length]))
          self._chunk_counter += 1
          first_unused_byte = pos + self._frame_length

        # Keep the remaining bytes for next time
        self._audio_buffer = audio_samples[first_unused_byte:]


# This actually executes in each worker thread!
class OnsetsTask(object):
  """Inference task."""

  def __init__(self, audio_chunk: AudioChunk):
    self.audio_chunk = audio_chunk
    self.result = None

  def __call__(self, model):
    samples = self.audio_chunk.samples[:, 0]
    self.result = model.infer(samples)
    self.timestep = model.get_timestep()


def result_collector(result_queue):
  """Collect and display results."""

  def notename(n, space):
    if space:
      return [' ', '  ', ' ', ' ', '  ', ' ', '  ', ' ', ' ', '  ', ' ',
              '  '][n % 12]
    return [
        Fore.BLUE + 'A' + Style.RESET_ALL,
        Fore.LIGHTBLUE_EX + 'A#' + Style.RESET_ALL,
        Fore.GREEN + 'B' + Style.RESET_ALL,
        Fore.CYAN + 'C' + Style.RESET_ALL,
        Fore.LIGHTCYAN_EX + 'C#' + Style.RESET_ALL,
        Fore.RED + 'D' + Style.RESET_ALL,
        Fore.LIGHTRED_EX + 'D#' + Style.RESET_ALL,
        Fore.YELLOW + 'E' + Style.RESET_ALL,
        Fore.WHITE + 'F' + Style.RESET_ALL,
        Fore.LIGHTBLACK_EX + 'F#' + Style.RESET_ALL,
        Fore.MAGENTA + 'G' + Style.RESET_ALL,
        Fore.LIGHTMAGENTA_EX + 'G#' + Style.RESET_ALL,
    ][n % 12]  #+ str(n//12)

  print('Listening to results..')
  # TODO(mtyka) Ensure serial stitching of results (no guarantee that
  # the blocks come in in order but they are all timestamped)
  while True:
    result = result_queue.get()
    serial = result.audio_chunk.serial
    result_roll = result.result
    if serial > 0:
      result_roll = result_roll[4:]
    for notes in result_roll:
      for i in range(6, len(notes) - 6):
        note = notes[i]
        is_frame = note[0] > 0.0
        notestr = notename(i, not is_frame)
        print(notestr, end='')
      print('|')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  results = multiprocessing.Queue()
  results_thread = threading.Thread(target=result_collector, args=(results,))
  results_thread.start()

  model = tflite_model.Model(model_path=FLAGS.model_path)
  overlap_timesteps = 4
  overlap_wav = model.get_hop_size(
  ) * overlap_timesteps + model.get_window_length()

  if FLAGS.wav_file:
    wav_data = open(FLAGS.wav_file, 'rb').read()
    samples = audio_recorder.wav_data_to_samples(wav_data,
                                                 model.get_sample_rate())
    samples = samples[:model.get_sample_rate() *
                      10]  # Only the first 10 seconds
    samples = samples.reshape((-1, 1))
    samples_length = samples.shape[0]
    # Extend samples with zeros
    samples = np.pad(
        samples, (0, model.get_input_wav_length()), mode='constant')
    for i, pos in enumerate(
        range(0, samples_length - model.get_input_wav_length() + overlap_wav,
              model.get_input_wav_length() - overlap_wav)):
      chunk = samples[pos:pos + model.get_input_wav_length()]
      task = OnsetsTask(AudioChunk(i, chunk))
      task(model)
      results.put(task)
  else:
    tasks = multiprocessing.JoinableQueue()

    ## Make and start the workers
    num_workers = 4
    workers = [
        TfLiteWorker(FLAGS.model_path, tasks, results)
        for i in range(num_workers)
    ]
    for w in workers:
      w.start()

    audio_feeder = AudioQueue(
        callback=lambda audio_chunk: tasks.put(OnsetsTask(audio_chunk)),
        audio_device_index=FLAGS.mic if FLAGS.mic is None else int(FLAGS.mic),
        sample_rate_hz=int(FLAGS.sample_rate_hz),
        model_sample_rate=model.get_sample_rate(),
        frame_length=model.get_input_wav_length(),
        overlap=overlap_wav)

    audio_feeder.start()


def console_entry_point():
  app.run(main)


if __name__ == '__main__':
  console_entry_point()
