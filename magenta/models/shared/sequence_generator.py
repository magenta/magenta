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

"""Abstract class for sequence generators.

Provides a uniform interface for interacting with generators for any model.
"""

import abc
import os
import tempfile

from note_seq.protobuf import generator_pb2
import tensorflow.compat.v1 as tf


class SequenceGeneratorError(Exception):  # pylint:disable=g-bad-exception-name
  """Generic exception for sequence generation errors."""
  pass


# TODO(adarob): Replace with tf.saver.checkpoint_file_exists when released.
def _checkpoint_file_exists(checkpoint_file_or_prefix):
  """Returns True if checkpoint file or files (for V2) exist."""
  return (tf.gfile.Exists(checkpoint_file_or_prefix) or
          tf.gfile.Exists(checkpoint_file_or_prefix + '.index'))


class BaseSequenceGenerator(object):
  """Abstract class for generators."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, model, details, checkpoint, bundle):
    """Constructs a BaseSequenceGenerator.

    Args:
      model: An instance of BaseModel.
      details: A generator_pb2.GeneratorDetails for this generator.
      checkpoint: Where to look for the most recent model checkpoint. Either a
          directory to be used with tf.train.latest_checkpoint or the path to a
          single checkpoint file. Or None if a bundle should be used.
      bundle: A generator_pb2.GeneratorBundle object that contains both a
          checkpoint and a metagraph. Or None if a checkpoint should be used.

    Raises:
      SequenceGeneratorError: if neither checkpoint nor bundle is set.
    """
    self._model = model
    self._details = details
    self._checkpoint = checkpoint
    self._bundle = bundle

    if self._checkpoint is None and self._bundle is None:
      raise SequenceGeneratorError(
          'Either checkpoint or bundle must be set')
    if self._checkpoint is not None and self._bundle is not None:
      raise SequenceGeneratorError(
          'Checkpoint and bundle cannot both be set')

    if self._bundle:
      if self._bundle.generator_details.id != self._details.id:
        raise SequenceGeneratorError(
            'Generator id in bundle (%s) does not match this generator\'s id '
            '(%s)' % (self._bundle.generator_details.id,
                      self._details.id))

    self._initialized = False

  @property
  def details(self):
    """Returns a GeneratorDetails description of this generator."""
    return self._details

  @property
  def bundle_details(self):
    """Returns the BundleDetails or None if checkpoint was used."""
    if self._bundle is None:
      return None
    return self._bundle.bundle_details

  @abc.abstractmethod
  def _generate(self, input_sequence, generator_options):
    """Implementation for sequence generation based on sequence and options.

    The implementation can assume that _initialize has been called before this
    method is called.

    Args:
      input_sequence: An input NoteSequence to base the generation on.
      generator_options: A GeneratorOptions proto with options to use for
          generation.
    Returns:
      The generated NoteSequence proto.
    """
    pass

  def initialize(self):
    """Builds the TF graph and loads the checkpoint.

    If the graph has already been initialized, this is a no-op.

    Raises:
      SequenceGeneratorError: If the checkpoint cannot be found.
    """
    if self._initialized:
      return

    # Either self._checkpoint or self._bundle should be set.
    # This is enforced by the constructor.
    if self._checkpoint is not None:
      # Check if the checkpoint file exists.
      if not _checkpoint_file_exists(self._checkpoint):
        raise SequenceGeneratorError(
            'Checkpoint path does not exist: %s' % (self._checkpoint))
      checkpoint_file = self._checkpoint
      # If this is a directory, try to determine the latest checkpoint in it.
      if tf.gfile.IsDirectory(checkpoint_file):
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_file)
      if checkpoint_file is None:
        raise SequenceGeneratorError(
            'No checkpoint file found in directory: %s' % self._checkpoint)
      if (not _checkpoint_file_exists(self._checkpoint) or
          tf.gfile.IsDirectory(checkpoint_file)):
        raise SequenceGeneratorError(
            'Checkpoint path is not a file: %s (supplied path: %s)' % (
                checkpoint_file, self._checkpoint))
      self._model.initialize_with_checkpoint(checkpoint_file)
    else:
      # Write checkpoint and metagraph files to a temp dir.
      tempdir = None
      try:
        tempdir = tempfile.mkdtemp()
        checkpoint_filename = os.path.join(tempdir, 'model.ckpt')
        with tf.gfile.Open(checkpoint_filename, 'wb') as f:
          # For now, we support only 1 checkpoint file.
          # If needed, we can later change this to support sharded checkpoints.
          f.write(self._bundle.checkpoint_file[0])
        metagraph_filename = os.path.join(tempdir, 'model.ckpt.meta')
        with tf.gfile.Open(metagraph_filename, 'wb') as f:
          f.write(self._bundle.metagraph_file)

        self._model.initialize_with_checkpoint_and_metagraph(
            checkpoint_filename, metagraph_filename)
      finally:
        # Clean up the temp dir.
        if tempdir is not None:
          tf.gfile.DeleteRecursively(tempdir)
    self._initialized = True

  def close(self):
    """Closes the TF session.

    If the session was already closed, this is a no-op.
    """
    if self._initialized:
      self._model.close()
      self._initialized = False

  def __enter__(self):
    """When used as a context manager, initializes the TF session."""
    self.initialize()
    return self

  def __exit__(self, *args):
    """When used as a context manager, closes the TF session."""
    self.close()

  def generate(self, input_sequence, generator_options):
    """Generates a sequence from the model based on sequence and options.

    Also initializes the TF graph if not yet initialized.

    Args:
      input_sequence: An input NoteSequence to base the generation on.
      generator_options: A GeneratorOptions proto with options to use for
          generation.

    Returns:
      The generated NoteSequence proto.
    """
    self.initialize()
    return self._generate(input_sequence, generator_options)

  def create_bundle_file(self, bundle_file, bundle_description=None):
    """Writes a generator_pb2.GeneratorBundle file in the specified location.

    Saves the checkpoint, metagraph, and generator id in one file.

    Args:
      bundle_file: Location to write the bundle file.
      bundle_description: A short, human-readable string description of this
          bundle.

    Raises:
      SequenceGeneratorError: if there is an error creating the bundle file.
    """
    if not bundle_file:
      raise SequenceGeneratorError('Bundle file location not specified.')
    if not self.details.id:
      raise SequenceGeneratorError(
          'Generator id must be included in GeneratorDetails when creating '
          'a bundle file.')

    if not self.details.description:
      tf.logging.warn('Writing bundle file with no generator description.')
    if not bundle_description:
      tf.logging.warn('Writing bundle file with no bundle description.')

    self.initialize()

    tempdir = None
    try:
      tempdir = tempfile.mkdtemp()
      checkpoint_filename = os.path.join(tempdir, 'model.ckpt')

      self._model.write_checkpoint_with_metagraph(checkpoint_filename)

      if not os.path.isfile(checkpoint_filename):
        raise SequenceGeneratorError(
            'Could not read checkpoint file: %s' % (checkpoint_filename))
      metagraph_filename = checkpoint_filename + '.meta'
      if not os.path.isfile(metagraph_filename):
        raise SequenceGeneratorError(
            'Could not read metagraph file: %s' % (metagraph_filename))

      bundle = generator_pb2.GeneratorBundle()
      bundle.generator_details.CopyFrom(self.details)
      if bundle_description:
        bundle.bundle_details.description = bundle_description
      with tf.gfile.Open(checkpoint_filename, 'rb') as f:
        bundle.checkpoint_file.append(f.read())
      with tf.gfile.Open(metagraph_filename, 'rb') as f:
        bundle.metagraph_file = f.read()

      with tf.gfile.Open(bundle_file, 'wb') as f:
        f.write(bundle.SerializeToString())
    finally:
      if tempdir is not None:
        tf.gfile.DeleteRecursively(tempdir)
