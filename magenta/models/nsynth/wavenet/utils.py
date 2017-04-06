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
"""A library of utility functions."""

# internal imports
import numpy as np
import scipy.io.wavfile
import tensorflow as tf


def save_wav(data, path):
  """This saves a wav file locally and then copies the result to CNS.

  Saving directly to CNS results in weird errors like:
  'colossus writes are append-only; cannot write to file offset 4,
  current length is 655404'

  Args:
    data: The float wav data, a 1D vector with values in [-1, 1].
    path: The CNS file path to which we save.
  """
  tmp_path = '/tmp/tmp-%d.wav' % np.random.randint(2**32)

  with tf.gfile.Open(tmp_path, 'w') as f:
    # The 0.9 is to prevent clipping.
    data_16bit = (0.9 * data) * 2**15
    scipy.io.wavfile.write(f, 16000, data_16bit.astype(np.int16))

  tf.gfile.Copy(tmp_path, path, overwrite=True)
  tf.gfile.Remove(tmp_path)


def load_wav(path):
  """Load a wav file and convert to floats within [-1, 1].

  Args:
    path: The CNS file path from which we load.

  Returns:
    The 16bit data in the range [-1, 1].
  """
  _, data_16bit = scipy.io.wavfile.read(tf.gfile.Open(path, 'r'))
  # Assert we are working with 16-bit audio.
  assert data_16bit.dtype == np.int16
  return data_16bit.astype(np.float32) / 2**15


def mu_law(x, mu=255):
  """A TF implementation of Mu-Law encoding.

  Args:
    x: The audio samples to encode.
    mu: The Mu to use in our Mu-Law.

  Returns:
    out: The Mu-Law encoded int8 data.
  """
  out = tf.sign(x) * tf.log(1 + mu * tf.abs(x)) / np.log(1 + mu)
  out = tf.cast(tf.floor(out * 128), tf.int8)
  return out


def inv_mu_law(x, mu=255):
  """A TF implementation of inverse Mu-Law.

  Args:
    x: The Mu-Law samples to decode.
    mu: The Mu we used to encode these samples.

  Returns:
    out: The decoded data.
  """
  x = tf.cast(x, tf.float32)
  out = (x + 0.5) * 2. / (mu + 1)
  out = tf.sign(out) / mu * ((1 + mu)**tf.abs(out) - 1)
  out = tf.where(tf.equal(x, 0), x, out)
  return out
