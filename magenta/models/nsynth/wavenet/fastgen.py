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

"""Utilities for "fast" wavenet generation with queues.

For more information, see:

Ramachandran, P., Le Paine, T., Khorrami, P., Babaeizadeh, M.,
Chang, S., Zhang, Y., ... Huang, T. (2017).
Fast Generation For Convolutional Autoregressive Models, 1-5.
"""
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet.h512_bo16 import Config
from magenta.models.nsynth.wavenet.h512_bo16 import FastGenerationConfig
import numpy as np
from scipy.io import wavfile
import tensorflow.compat.v1 as tf


def sample_categorical(probability_mass_function):
  """Sample from a categorical distribution.

  Args:
    probability_mass_function: Output of a softmax over categories.
      Array of shape [batch_size, number of categories]. Rows sum to 1.

  Returns:
    idxs: Array of size [batch_size, 1]. Integer of category sampled.
  """
  if probability_mass_function.ndim == 1:
    probability_mass_function = np.expand_dims(probability_mass_function, 0)
  batch_size = probability_mass_function.shape[0]
  cumulative_density_function = np.cumsum(probability_mass_function, axis=1)
  rand_vals = np.random.rand(batch_size)
  idxs = np.zeros([batch_size, 1])
  for i in range(batch_size):
    idxs[i] = cumulative_density_function[i].searchsorted(rand_vals[i])
  return idxs


def load_nsynth(batch_size=1, sample_length=64000):
  """Load the NSynth autoencoder network.

  Args:
    batch_size: Batch size number of observations to process. [1]
    sample_length: Number of samples in the input audio. [64000]
  Returns:
    graph: The network as a dict with input placeholder in {"X"}
  """
  config = Config()
  with tf.device("/gpu:0"):
    x = tf.placeholder(tf.float32, shape=[batch_size, sample_length])
    graph = config.build({"wav": x}, is_training=False)
    graph.update({"X": x})
  return graph


def load_fastgen_nsynth(batch_size=1):
  """Load the NSynth fast generation network.

  Args:
    batch_size: Batch size number of observations to process. [1]
  Returns:
    graph: The network as a dict with input placeholder in {"X"}
  """
  config = FastGenerationConfig(batch_size=batch_size)
  with tf.device("/gpu:0"):
    x = tf.placeholder(tf.float32, shape=[batch_size, 1])
    graph = config.build({"wav": x})
    graph.update({"X": x})
  return graph


def encode(wav_data, checkpoint_path, sample_length=64000):
  """Generate an array of encodings from an array of audio.

  Args:
    wav_data: Numpy array [batch_size, sample_length]
    checkpoint_path: Location of the pretrained model.
    sample_length: The total length of the final wave file, padded with 0s.
  Returns:
    encoding: a [mb, 125, 16] encoding (for 64000 sample audio file).
  """
  if wav_data.ndim == 1:
    wav_data = np.expand_dims(wav_data, 0)
  batch_size = wav_data.shape[0]

  # Load up the model for encoding and find the encoding of "wav_data"
  session_config = tf.ConfigProto(allow_soft_placement=True)
  session_config.gpu_options.allow_growth = True
  with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
    hop_length = Config().ae_hop_length
    wav_data, sample_length = utils.trim_for_encoding(wav_data, sample_length,
                                                      hop_length)
    net = load_nsynth(batch_size=batch_size, sample_length=sample_length)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
    encodings = sess.run(net["encoding"], feed_dict={net["X"]: wav_data})
  return encodings


def load_batch_audio(files, sample_length=64000):
  """Load a batch of audio from either .wav files.

  Args:
    files: A list of filepaths to .wav files.
    sample_length: Maximum sample length

  Returns:
    batch: A padded array of audio [n_files, sample_length]
  """
  batch = []
  # Load the data
  for f in files:
    data = utils.load_audio(f, sample_length, sr=16000)
    length = data.shape[0]
    # Add padding if less than sample length
    if length < sample_length:
      padded = np.zeros([sample_length])
      padded[:length] = data
      batch.append(padded)
    else:
      batch.append(data)
  # Return as an numpy array
  batch = np.array(batch)
  return batch


def load_batch_encodings(files, sample_length=125):
  """Load a batch of encodings from .npy files.

  Args:
    files: A list of filepaths to .npy files
    sample_length: Maximum sample length

  Raises:
    ValueError: .npy array has wrong dimensions.

  Returns:
    batch: A padded array encodings [batch, length, dims]
  """
  batch = []
  # Load the data
  for f in files:
    data = np.load(f)
    if data.ndim != 2:
      raise ValueError("Encoding file should have 2 dims "
                       "[time, channels], not {}".format(data.ndim))
    length, channels = data.shape
    # Add padding or crop if not equal to sample length
    if length < sample_length:
      padded = np.zeros([sample_length, channels])
      padded[:length, :] = data
      batch.append(padded)
    else:
      batch.append(data[:sample_length])
  # Return as an numpy array
  batch = np.array(batch)
  return batch


def save_batch(batch_audio, batch_save_paths):
  for audio, name in zip(batch_audio, batch_save_paths):
    tf.logging.info("Saving: %s" % name)
    wavfile.write(name, 16000, audio)


def generate_audio_sample(sess, net, audio, encoding):
  """Generate a single sample of audio from an encoding.

  Args:
    sess: tf.Session to use.
    net: Loaded wavenet network (dictionary of endpoint tensors).
    audio: Previously generated audio [batch_size, 1].
    encoding: Encoding at current time index [batch_size, dim].

  Returns:
    audio_gen: Generated audio [batch_size, 1]
  """
  probability_mass_function = sess.run(
      [net["predictions"], net["push_ops"]],
      feed_dict={net["X"]: audio, net["encoding"]: encoding})[0]
  sample_bin = sample_categorical(probability_mass_function)
  audio_gen = utils.inv_mu_law_numpy(sample_bin - 128)
  return audio_gen


def synthesize(encodings,
               save_paths,
               checkpoint_path="model.ckpt-200000",
               samples_per_save=10000):
  """Synthesize audio from an array of encodings.

  Args:
    encodings: Numpy array with shape [batch_size, time, dim].
    save_paths: Iterable of output file names.
    checkpoint_path: Location of the pretrained model. [model.ckpt-200000]
    samples_per_save: Save files after every amount of generated samples.
  """
  session_config = tf.ConfigProto(allow_soft_placement=True)
  session_config.gpu_options.allow_growth = True
  with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
    net = load_fastgen_nsynth(batch_size=encodings.shape[0])
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    # Get lengths
    batch_size, encoding_length, _ = encodings.shape
    hop_length = Config().ae_hop_length
    total_length = encoding_length * hop_length

    # initialize queues w/ 0s
    sess.run(net["init_ops"])

    # Regenerate the audio file sample by sample
    audio_batch = np.zeros(
        (batch_size, total_length), dtype=np.float32)
    audio = np.zeros([batch_size, 1])

    for sample_i in range(total_length):
      encoding_i = sample_i // hop_length
      audio = generate_audio_sample(sess, net,
                                    audio, encodings[:, encoding_i, :])
      audio_batch[:, sample_i] = audio[:, 0]
      if sample_i % 100 == 0:
        tf.logging.info("Sample: %d" % sample_i)
      if sample_i % samples_per_save == 0 and save_paths:
        save_batch(audio_batch, save_paths)

    save_batch(audio_batch, save_paths)
