# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Utilities for "fast" wavenet generation with queues.

For more information, see:

Ramachandran, P., Le Paine, T., Khorrami, P., Babaeizadeh, M.,
Chang, S., Zhang, Y., ... Huang, T. (2017).
Fast Generation For Convolutional Autoregressive Models, 1-5.
"""
import os
import numpy as np
from scipy.io import wavfile
import tensorflow as tf

from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet.h512_bo16 import Config
from magenta.models.nsynth.wavenet.h512_bo16 import FastGenerationConfig


def sample_categorical(pmf):
  """Sample from a categorical distribution.

  Args:
    pmf: Probablity mass function. Output of a softmax over categories.
      Array of shape [batch_size, number of categories]. Rows sum to 1.

  Returns:
    idxs: Array of size [batch_size, 1]. Integer of category sampled.
  """
  if pmf.ndim == 1:
    pmf = np.expand_dims(pmf, 0)
  batch_size = pmf.shape[0]
  cdf = np.cumsum(pmf, axis=1)
  rand_vals = np.random.rand(batch_size)
  idxs = np.zeros([batch_size, 1])
  for i in range(batch_size):
    idxs[i] = cdf[i].searchsorted(rand_vals[i])
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
  """Generate an array of embeddings from an array of audio.

  Args:
    wav_data: Numpy array [batch_size, sample_length]
    checkpoint_path: Location of the pretrained model.
    sample_length: The total length of the final wave file, padded with 0s.
  Returns:
    encoding: a [mb, 125, 16] encoding (for 64000 sample audio file).
  """
  if wav_data.ndim == 1:
    wav_data = np.expand_dims(wav_data, 0)
    batch_size = 1
  elif wav_data.ndim == 2:
    batch_size = wav_data.shape[0]

  # Load up the model for encoding and find the encoding of "wav_data"
  session_config = tf.ConfigProto(allow_soft_placement=True)
  with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
    hop_length = Config().ae_hop_length
    wav_data, sample_length = utils.trim_for_encoding(wav_data, sample_length,
                                                      hop_length)
    net = load_nsynth(batch_size=batch_size, sample_length=sample_length)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
    encodings = sess.run(net["encoding"], feed_dict={net["X"]: wav_data})
  return encodings


def load_batch(files, sample_length=64000):
  """Load a batch of data from either .wav or .npy files.

  Args:
    files: A list of filepaths to .wav or .npy files
    sample_length: Maximum sample length

  Returns:
    batch_data: A padded array of audio or embeddings [batch, length, (dims)]
  """
  batch_data = []
  max_length = 0
  is_npy = (os.path.splitext(files[0])[1] == ".npy")
  # Load the data
  for f in files:
    if is_npy:
      data = np.load(f)
      batch_data.append(data)
    else:
      data = utils.load_audio(f, sample_length, sr=16000)
      batch_data.append(data)
    if data.shape[0] > max_length:
      max_length = data.shape[0]
  # Add padding
  for i, data in enumerate(batch_data):
    if data.shape[0] < max_length:
      if is_npy:
        padded = np.zeros([max_length, +data.shape[1]])
        padded[:data.shape[0], :] = data
      else:
        padded = np.zeros([max_length])
        padded[:data.shape[0]] = data
      batch_data[i] = padded
  # Return arrays
  batch_data = np.array(batch_data)
  return batch_data


def save_batch(batch_audio, batch_save_paths):
  for audio, name in zip(batch_audio, batch_save_paths):
    tf.logging.info("Saving: %s" % name)
    wavfile.write(name, 16000, audio)


def synthesize(encodings,
               save_paths,
               checkpoint_path="model.ckpt-200000",
               samples_per_save=1000):
  """Synthesize audio from an array of embeddings.

  Args:
    encodings: Numpy array with shape [batch_size, time, dim].
    save_paths: Iterable of output file names.
    checkpoint_path: Location of the pretrained model. [model.ckpt-200000]
    samples_per_save: Save files after every amount of generated samples.
  """
  hop_length = Config().ae_hop_length
  # Get lengths
  batch_size = encodings.shape[0]
  encoding_length = encodings.shape[1]
  total_length = encoding_length * hop_length

  session_config = tf.ConfigProto(allow_soft_placement=True)
  with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
    net = load_fastgen_nsynth(batch_size=batch_size)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    # initialize queues w/ 0s
    sess.run(net["init_ops"])

    # Regenerate the audio file sample by sample
    audio_batch = np.zeros((batch_size, total_length,), dtype=np.float32)
    audio = np.zeros([batch_size, 1])

    for sample_i in range(total_length):
      enc_i = sample_i // hop_length
      pmf = sess.run(
          [net["predictions"], net["push_ops"]],
          feed_dict={net["X"]: audio,
                     net["encoding"]: encodings[:, enc_i, :]})[0]
      sample_bin = sample_categorical(pmf)
      audio = utils.inv_mu_law_numpy(sample_bin - 128)
      audio_batch[:, sample_i] = audio[:, 0]
      if sample_i % 100 == 0:
        tf.logging.info("Sample: %d" % sample_i)
      if sample_i % samples_per_save == 0:
        save_batch(audio_batch, save_paths)
  save_batch(audio_batch, save_paths)
