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
"""Code for interacting with the TFLite model."""
import importlib
import numpy as np

try:
  tflite = importlib.import_module('tensorflow.compat.v1').lite
except ModuleNotFoundError:
  try:
    tflite = importlib.import_module('tflite_runtime.interpreter')
  except ModuleNotFoundError:
    print('Either Tensorflow or tflite_runtime must be installed.')
    raise


def get_model_detail(model_path):
  interpreter = tflite.Interpreter(model_path=model_path)
  return (interpreter.get_input_details(), interpreter.get_output_details())


class Model(object):
  """Manage Onsets and Frames TFLite model."""
  # Model specific constants.
  MODEL_SAMPLE_RATE = 16000
  MODEL_WINDOW_LENGTH = 2048

  def __init__(self, model_path):
    self._interpreter = tflite.Interpreter(model_path=model_path)
    self._interpreter.allocate_tensors()
    self._input_details = self._interpreter.get_input_details()
    self._output_details = self._interpreter.get_output_details()
    self._output_index = {
        detail['name']: detail['index'] for detail in self._output_details
    }
    self._input_wav_length = self._input_details[0]['shape'][0]
    self._output_roll_length = self._output_details[0]['shape'][1]
    assert (self._input_wav_length -
            Model.MODEL_WINDOW_LENGTH) % (self._output_roll_length - 1) == 0
    self._hop_size = (self._input_wav_length - Model.MODEL_WINDOW_LENGTH) // (
        self._output_roll_length - 1)
    self._timestep = float(self._hop_size) / Model.MODEL_SAMPLE_RATE

  def get_sample_rate(self):
    return Model.MODEL_SAMPLE_RATE

  def get_window_length(self):
    return Model.MODEL_WINDOW_LENGTH

  def get_timestep(self):
    """Returns the clock time represented by each output slice in ms."""
    return int(1000 * self._timestep)

  def get_input_wav_length(self):
    return self._input_wav_length

  def get_hop_size(self):
    return self._hop_size

  def infer(self, samples):
    """Do inference over the provided samples."""
    self._interpreter.set_tensor(
        self._interpreter.get_input_details()[0]['index'],
        samples)

    self._interpreter.invoke()
    predictions = [
        self._interpreter.get_tensor(self._output_index['frame_logits']),
        self._interpreter.get_tensor(self._output_index['onset_logits']),
        self._interpreter.get_tensor(self._output_index['offset_logits']),
        self._interpreter.get_tensor(self._output_index['velocity_values']),
    ]
    # Use axis 0 to stack but then shove the axis to the end so its
    # [time, midinote (0-88), {frame,onset, offset, velocity}]
    # The first 3 are logits, the last is simply the velocity
    result = np.transpose(np.concatenate(predictions, axis=0), [1, 2, 0])

    # TODO(fjord): Add onset filtering similar to our regular inference
    # setup.
    return result
