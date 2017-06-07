import tensorflow as tf
from scipy.io import wavfile
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet.h512_bo16 import Config, FastGenerationConfig
import numpy as np


def inv_mu_law(x, mu=255.0):
  """A numpy implementation of inverse Mu-Law.
  Args:
    x: The Mu-Law samples to decode.
    mu: The Mu we used to encode these samples.
  Returns:
    out: The decoded data.
  """
  x = np.array(x).astype(np.float32)
  out = (x + 0.5) * 2. / (mu + 1)
  out = np.sign(out) / mu * ((1 + mu)**np.abs(out) - 1)
  out = np.where(np.equal(x, 0), x, out)
  return out


def load_audio(wav_file, sample_length=64000):
  """Padded loading of a wave file.
  Args:
    wav_file: Location of a wave file to load.
    sample_length: The total length of the final wave file, padded with 0s.
  Returns:
    out: The padded audio in samples from -1.0 to 1.0
  """
  wav_data = np.array([utils.load_wav(wav_file)[:sample_length]])
  wav_data_padded = np.zeros((1, sample_length))
  wav_data_padded[0, :min(sample_length, wav_data.shape[1])] = wav_data
  wav_data = wav_data_padded
  return wav_data


def load_nsynth(batch_size=1, sample_length=64000):
  """Load the NSynth autoencoder network.
  Args:
    batch_size: Batch size number of observations to process. [1]
    sample_length: Number of samples in the input audio. [64000]
  Returns:
    graph: The network as a dict with input placeholder in {'X'}
  """
  config = Config()
  with tf.device('/gpu:0'):
    X = tf.placeholder(
      tf.float32, shape=[batch_size, sample_length])
    graph = config.build({"wav": X}, is_training=False)
    graph.update({'X': X})
  return graph


def load_fastgen_nsynth(batch_size=1, sample_length=64000):
  """Load the NSynth fast generation network.
  Args:
    batch_size: Batch size number of observations to process. [1]
    sample_length: Number of samples in the input audio. [64000]
  Returns:
    graph: The network as a dict with input placeholder in {'X'}
  """
  config = FastGenerationConfig()
  X = tf.placeholder(
    tf.float32, shape=[batch_size, 1])
  graph = config.build({"wav": X})
  graph.update({'X': X})
  return graph


def encode(wav_file, ckpt_path, sample_length=64000):
  """Padded loading of a wave file.
  Args:
    wav_file: Location of a wave file to load.
    ckpt_path: Location of the pretrained model.
    sample_length: The total length of the final wave file, padded with 0s.
  Returns:
    encoding: a [mb, 125, 16] encoding (for 64000 sample audio file).
  """

  # Audio to resynthesize
  wav_data = load_audio(wav_file, sample_length)

  # Load up the model for encoding and find the encoding of 'wav_data'
  with tf.Graph().as_default(), tf.Session(
          config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    net = load_nsynth()
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)
    encoding = sess.run(net['encoding'], feed_dict={
      net['X']: wav_data})[0]

  return encoding


def synthesize(wav_file,
               ckpt_path='model.ckpt-200000',
               out_file='synthesis.wav',
               sample_length=64000,
               synth_length=64000):
  """Resynthesize an input audio file.

  Args:
    wav_file: Location of a wave file to load.
    ckpt_path: Location of the pretrained model. [model.ckpt-200000]
    out_file: Location to save the synthesized wave file. [synthesis.wav]
    sample_length: The total length of the input wave file, padded with 0s. [64000]
    synth_length: The total length to synthesize. [64000]
  """

  # Load up the model for encoding and find the encoding
  encoding = encode(wav_file, ckpt_path)
  encoding_length = encoding.shape[0]

  with tf.Graph().as_default(), tf.Session(
          config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    net = load_fastgen_nsynth()
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)

    # initialize queues w/ 0s
    sess.run(net['init_ops'])

    # Regenerate the audio file sample by sample
    wav_synth = np.zeros((sample_length,),
        dtype=np.float32)
    audio = np.float32(0)

    for sample_i in range(synth_length):
      enc_i = int(sample_i /
        float(sample_length) *
        float(encoding_length))
      res = sess.run(
          [net['predictions'], net['push_ops']],
          feed_dict={
            net['X']: np.atleast_2d(audio),
            net['encoding']: encoding[enc_i]})[0]
      cdf = np.cumsum(res)
      idx = np.random.rand()
      i = 0
      while(cdf[i] < idx):
        i = i + 1
      audio = inv_mu_law(i - 128)
      wav_synth[sample_i] = audio

  wavfile.write(out_file, 16000, wav_synth)
