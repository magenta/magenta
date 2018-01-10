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
"""A class for sampling, encoding, and decoding from trained MusicVAE models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

# internal imports
import numpy as np
import tensorflow as tf


class NoExtractedExamplesException(Exception):
  pass


class MultipleExtractedExamplesException(Exception):
  pass


class TrainedModel(object):
  """An interface to a trained model for encoding, decoding, and sampling.

  Args:
    config: The Config to build the model graph with.
    batch_size: The batch size to build the model graph with.
    checkpoint_dir_or_path: The directory containing checkpoints for the model,
      the most recent of which will be loaded, or a direct path to a specific
      checkpoint.
    session_target: Optinal execution engine to connect to. Defaults to
      in-process.
    sample_kwargs: Additional, non-tensor keyword arguments to pass to sample
      call.
  """

  def __init__(self, config, batch_size, checkpoint_dir_or_path=None,
               session_target='', **sample_kwargs):
    checkpoint_path = (tf.train.latest_checkpoint(checkpoint_dir_or_path)
                       if tf.gfile.IsDirectory(checkpoint_dir_or_path) else
                       checkpoint_dir_or_path)
    self._config = copy.deepcopy(config)
    self._config.hparams.batch_size = batch_size
    with tf.Graph().as_default():
      model = self._config.model
      model.build(
          self._config.hparams,
          self._config.note_sequence_converter.output_depth,
          is_training=False)
      # Input placeholders
      self._temperature = tf.placeholder(tf.float32, shape=())
      self._z_input = (
          tf.placeholder(tf.float32,
                         shape=[batch_size, self._config.hparams.z_size])
          if self._config.hparams.conditional else None)
      self._inputs = tf.placeholder(
          tf.float32,
          shape=[batch_size, None,
                 self._config.note_sequence_converter.input_depth])
      self._inputs_length = tf.placeholder(tf.int32, shape=[batch_size])
      self._max_length = tf.placeholder(tf.int32, shape=())
      # Outputs
      self._outputs = model.sample(
          batch_size,
          max_length=self._max_length,
          z=self._z_input,
          temperature=self._temperature,
          **sample_kwargs)
      if self._config.hparams.conditional:
        q_z = model.encode(self._inputs, self._inputs_length)
        self._mu = q_z.loc
        self._sigma = q_z.scale.diag
        self._z = q_z.sample()
      # Restore graph
      self._sess = tf.Session(target=session_target)
      saver = tf.train.Saver()
      saver.restore(self._sess, checkpoint_path)

  def sample(self, n=None, length=None, temperature=1.0, same_z=False):
    """Generates random samples from the model.

    Args:
      n: The number of samples to return. A full batch will be returned if not
        specified.
      length: The maximum length of a sample in decoder iterations. Required
        if end tokens are not being used.
      temperature: The softmax temperature to use (if applicable).
      same_z: Whether to use the same latent vector for all samples in the
        batch (if applicable).
    Returns:
      A list of samples as NoteSequence objects.
    Raises:
      ValueError: If `length` is not specified and an end token is not being
        used.
    """
    batch_size = self._config.hparams.batch_size
    n = n or batch_size
    z_size = self._config.hparams.z_size

    if not length and self._config.note_sequence_converter.end_token is None:
      raise ValueError(
          'A length must be specified when the end token is not used.')
    length = length or tf.int32.max

    feed_dict = {
        self._temperature: temperature,
        self._max_length: length
    }

    if self._z_input is not None:
      if same_z:
        z = np.random.randn(z_size).astype(np.float32)
        z = np.tile(z, (batch_size, 1))
      else:
        z = np.random.randn(batch_size, z_size).astype(np.float32)
      feed_dict[self._z_input] = z

    outputs = []
    for _ in range(int(np.ceil(n / batch_size))):
      outputs.append(self._sess.run(self._outputs, feed_dict))
    return self._config.note_sequence_converter.to_notesequences(
        np.vstack(outputs)[:n])

  def encode(self, note_sequences, assert_same_length=False):
    """Encodes a collection of NoteSequences into latent vectors.

    Args:
      note_sequences: A collection of NoteSequence objects to encode.
      assert_same_length: Whether to raise an AssertionError if all of the
        extracted sequences are not the same length.
    Returns:
      The encoded `z`, `mu`, and `sigma` values.
    Raises:
      RuntimeError: If called for a non-conditional model.
      NoExtractedExamplesException: If no examples were extracted.
      MultipleExtractedExamplesException: If multiple examples were extracted.
      AssertionError: If `assert_same_length` is True and any extracted
        sequences differ in length.
    """
    if not self._config.hparams.conditional:
      raise RuntimeError('Cannot encode with a non-conditional model.')

    inputs = []
    for note_sequence in note_sequences:
      extracted_inputs, _ = self._config.note_sequence_converter.to_tensors(
          note_sequence)
      if not extracted_inputs:
        raise NoExtractedExamplesException(
            'No examples extracted from NoteSequence: %s' % note_sequence)
      if len(extracted_inputs) > 1:
        raise MultipleExtractedExamplesException(
            'Multiple (%d) examples extracted from NoteSequence: %s' %
            (len(extracted_inputs), note_sequence))
      inputs.append(extracted_inputs[0])
      if assert_same_length and len(inputs[0]) != len(inputs[-1]):
        raise AssertionError(
            'Sequences 0 and %d have different lengths: %d vs %d' %
            (len(inputs) - 1, len(inputs[0]), len(inputs[-1])))
    return self.encode_tensors(inputs)

  def encode_tensors(self, input_tensors):
    """Encodes a collection of input tensors into latent vectors.

    Args:
      input_tensors: Collection of input tensors to encode.
    Returns:
      The encoded `z`, `mu`, and `sigma` values.
    Raises:
       RuntimeError: If called for a non-conditional model.
    """
    if not self._config.hparams.conditional:
      raise RuntimeError('Cannot encode with a non-conditional model.')

    n = len(input_tensors)
    input_depth = self._config.note_sequence_converter.input_depth
    batch_size = self._config.hparams.batch_size

    batch_pad_amt = -n % batch_size
    input_tensors += [np.zeros([0, input_depth])] * batch_pad_amt

    inputs_length = np.array([len(t) for t in input_tensors], np.int32)
    inputs_array = np.zeros(
        [len(input_tensors), max(inputs_length), input_depth])
    for i in range(len(input_tensors)):
      inputs_array[i, :inputs_length[i]] = input_tensors[i]

    outputs = []
    for i in range(len(inputs_array) // batch_size):
      batch_begin = i * batch_size
      batch_end = (i+1) * batch_size
      feed_dict = {self._inputs: inputs_array[batch_begin:batch_end],
                   self._inputs_length: inputs_length[batch_begin:batch_end]}
      outputs.append(
          self._sess.run([self._z, self._mu, self._sigma], feed_dict))
    assert outputs
    return tuple(np.vstack(v)[:n] for v in zip(*outputs))

  def decode(self, z, length=None, temperature=1.0):
    """Decodes a collection of latent vectors into NoteSequences.

    Args:
      z: A collection of latent vectors to decode.
      length: The maximum length of a sample in decoder iterations. Required
        if end tokens are not being used.
      temperature: The softmax temperature to use (if applicable).
    Returns:
      A list of decodings as NoteSequence objects.
    Raises:
      RuntimeError: If called for a non-conditional model.
      ValueError: If `length` is not specified and an end token is not being
        used.
    """
    return self._config.note_sequence_converter.to_notesequences(
        self.decode_to_tensors(z, length, temperature))

  def decode_to_tensors(self, z, length=None, temperature=1.0):
    """Decodes a collection of latent vectors into output tensors.

    Args:
      z: A collection of latent vectors to decode.
      length: The maximum length of a sample in decoder iterations. Required
        if end tokens are not being used.
      temperature: The softmax temperature to use (if applicable).
    Returns:
      Outputs from decoder as a 2D numpy array.
    Raises:
      RuntimeError: If called for a non-conditional model.
      ValueError: If `length` is not specified and an end token is not being
        used.
    """
    if not self._config.hparams.conditional:
      raise RuntimeError('Cannot decode with a non-conditional model.')

    if not length and self._config.note_sequence_converter.end_token is None:
      raise ValueError(
          'A length must be specified when the end token is not used.')
    batch_size = self._config.hparams.batch_size
    n = len(z)
    length = length or tf.int32.max

    batch_pad_amt = -n % batch_size
    z = np.pad(z, [(0, batch_pad_amt), (0, 0)], mode='constant')

    outputs = []
    for i in range(len(z) // batch_size):
      feed_dict = {
          self._temperature: temperature,
          self._z_input: z[i*batch_size:(i+1)*batch_size],
          self._max_length: length,
      }
      outputs.extend(self._sess.run(self._outputs, feed_dict))
    return outputs[:n]
