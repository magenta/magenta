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
import numpy as np
from scipy.io import wavfile
import tensorflow as tf

from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet.h512_bo16 import Config
from magenta.models.nsynth.wavenet.h512_bo16 import FastGenerationConfig


def sample_categorical(pmf):
  cdf = np.cumsum(pmf)
  idx = np.random.rand()
  i = cdf.searchsorted(idx)
  return i


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
  """Padded loading of a wave file.

  Args:
    wav_data: Numpy array [batch_size, sample_length]
    checkpoint_path: Location of the pretrained model.
    sample_length: The total length of the final wave file, padded with 0s.
  Returns:
    encoding: a [mb, 125, 16] encoding (for 64000 sample audio file).
    hop_length: Pooling size of the autoencoder.
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
    wav_data, sample_length = utils.trim_for_encoding(wav_data,
                                                      sample_length,
                                                      hop_length)
    net = load_nsynth(batch_size=batch_size,
                      sample_length=sample_length)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
    encoding = sess.run(net["encoding"], feed_dict={net["X"]: wav_data})
  return encoding, hop_length


def synthesize(source_file,
               checkpoint_path="model.ckpt-200000",
               out_file="synthesis.wav",
               sample_length=64000,
               samples_per_save=1000):
  """Resynthesize an input audio file.

  Args:
    source_file: Location of a wave or .npy file to load.
    checkpoint_path: Location of the pretrained model. [model.ckpt-200000]
    out_file: Location to save the synthesized wave file. [synthesis.wav]
    sample_length: Length of file to synthesize. [source_file.length]
    samples_per_save: Save a .wav after every amount of samples.

  Raises:
    RuntimeError: Source_file should be .wav or .npy.
  """
  if source_file.endswith(".npy"):
    encoding = np.load(source_file)
    hop_length = Config().ae_hop_length
  elif source_file.endswith(".wav"):
    # Audio to resynthesize
    wav_data = utils.load_audio(source_file, sample_length, sr=16000)
    # Load up the model for encoding and find the encoding
    encoding, hop_length = encode(wav_data,
                                  checkpoint_path, sample_length=sample_length)
    if encoding.ndim == 3:
      encoding = encoding[0]
  else:
    raise RuntimeError("File must be .wav or .npy")
  # Get lengths
  encoding_length = encoding.shape[0]
  total_length = encoding_length * hop_length

  session_config = tf.ConfigProto(allow_soft_placement=True)
  with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
    net = load_fastgen_nsynth()
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    # initialize queues w/ 0s
    sess.run(net["init_ops"])

    # Regenerate the audio file sample by sample
    wav_synth = np.zeros((total_length,), dtype=np.float32)
    audio = np.float32(0)

    for sample_i in range(total_length):
      enc_i = sample_i // hop_length
      pmf = sess.run([net["predictions"], net["push_ops"]],
                     feed_dict={net["X"]: np.atleast_2d(audio),
                                net["encoding"]: encoding[enc_i]})[0]
      sample_bin = sample_categorical(pmf)
      audio = utils.inv_mu_law_numpy(sample_bin - 128)
      wav_synth[sample_i] = audio
      if sample_i % 100 == 0:
        tf.logging.info("Sample: %d" % sample_i)
      if sample_i % samples_per_save == 0:
        wavfile.write(out_file, 16000, wav_synth)

  wavfile.write(out_file, 16000, wav_synth)
