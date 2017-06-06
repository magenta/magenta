"""Utilities for "fast" wavenet generation with queues.

For more information, see:

Ramachandran, P., Le Paine, T., Khorrami, P., Babaeizadeh, M.,
Chang, S., Zhang, Y., ... Huang, T. (2017).
Fast Generation For Convolutional Autoregressive Models, 1-5.
"""
import tensorflow as tf
from scipy.io import wavfile
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet.h512_bo16 import Config, FastGenerationConfig
import numpy as np

def sample_categorical(pmf):
  cdf = np.cumsum(pmf)
  idx = np.random.rand()
  i = cdf.searchsorted(idx)
  return i


def load_nsynth(encoding=True, batch_size=1, sample_length=64000):
  """Load the NSynth autoencoder network.
  Args:
    encoding: Load the full autoencoder or else just the decoder. [True]
    batch_size: Batch size number of observations to process. [1]
    sample_length: Number of samples in the input audio. [64000]
  Returns:
    graph: The network as a dict with input placeholder in {'X'}
  """
  config = Config()
  with tf.device('/gpu:0'):
    X = tf.placeholder(tf.float32, shape=[batch_size, sample_length])
    graph = config.build({"wav": X}, is_training=False)
    graph.update({'X': X})
  return graph


def load_fastgen_nsynth(batch_size=1):
  """Load the NSynth fast generation network.
  Args:
    batch_size: Batch size number of observations to process. [1]
  Returns:
    graph: The network as a dict with input placeholder in {'X'}
  """
  config = FastGenerationConfig()
  with tf.device('/gpu:0'):
    X = tf.placeholder(tf.float32, shape=[batch_size, 1])
    graph = config.build({"wav": X})
    graph.update({'X': X})
  return graph


def encode(wav_data, ckpt_path, sample_length=64000):
  """Padded loading of a wave file.
  Args:
    wav_file: Location of a wave file to load.
    ckpt_path: Location of the pretrained model.
    sample_length: The total length of the final wave file, padded with 0s.
  Returns:
    encoding: a [mb, 125, 16] encoding (for 64000 sample audio file).
  """
  if wav_data.ndim == 1:
    wav_data = np.expand_dims(wav_data, 0)

  # Load up the model for encoding and find the encoding of 'wav_data'
  session_config = tf.ConfigProto(allow_soft_placement=True)
  with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
    net = load_nsynth(encoding=True, sample_length=sample_length)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)
    encoding = sess.run(net['encoding'], feed_dict={
      net['X']: wav_data})[0]
  return encoding


def synthesize(wav_file,
               ckpt_path='model.ckpt-200000',
               out_file='synthesis.wav',
               sample_length=64000,
               samples_per_save=1000):
  """Resynthesize an input audio file.

  Args:
    wav_file: Location of a wave file to load.
    ckpt_path: Location of the pretrained model. [model.ckpt-200000]
    out_file: Location to save the synthesized wave file. [synthesis.wav]
    sample_length: Length of file to synthesize. [wav_file.length]
  """
  # Audio to resynthesize
  wav_data = utils.load_audio(wav_file, sample_length, sr=16000)

  print "WAVE", wav_data.shape
  print "SL", sample_length


  # Make sure sample_length is a multiple of hop_length
  hop_length = 512
  if sample_length > wav_data.size:
    sample_length = wav_data.size
  sample_length = (sample_length // hop_length) * hop_length
  wav_data = wav_data[:sample_length]

  print "WAVE", wav_data.shape
  print "SL", sample_length

  # Load up the model for encoding and find the encoding
  encoding = encode(wav_data, ckpt_path, sample_length=sample_length)
  encoding_length = encoding.shape[0]

  session_config = tf.ConfigProto(allow_soft_placement=True)
  with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
    net = load_fastgen_nsynth()
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)

    # initialize queues w/ 0s
    sess.run(net['init_ops'])

    # Regenerate the audio file sample by sample
    wav_synth = np.zeros((sample_length,),
        dtype=np.float32)
    audio = np.float32(0)

    for sample_i in range(sample_length):
      enc_i = int(sample_i / float(sample_length) * float(encoding_length))
      pmf = sess.run([net['predictions'], net['push_ops']],
              feed_dict={net['X']: np.atleast_2d(audio),
                         net['encoding']: encoding[enc_i]})[0]
      sample_bin = sample_categorical(pmf)
      audio = utils.inv_mu_law_numpy(sample_bin - 128)
      wav_synth[sample_i] = audio
      if sample_i % 100 == 0:
        tf.logging.info("Sample: %d" % sample_i)
      if sample_i % samples_per_save == 0:
        wavfile.write(out_file, 16000, wav_synth)

  wavfile.write(out_file, 16000, wav_synth)
