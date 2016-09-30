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
"""Abstract class for sequence generators.

Provides a uniform interface for interacting with generators for any model.
"""

import abc
import os
import tempfile

# internal imports

import tensorflow as tf

from magenta.protobuf import generator_pb2


class SequenceGeneratorException(Exception):
  """Generic exception for sequence generation errors."""
  pass


class BaseSequenceGenerator(object):
  """Abstract class for generators."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, details, checkpoint, bundle):
    """Constructs a BaseSequenceGenerator.

    Args:
      details: A generator_pb2.GeneratorDetails for this generator.
      checkpoint: Where to look for the most recent model checkpoint. Either a
          directory to be used with tf.train.latest_checkpoint or the path to a
          single checkpoint file. Or None if a bundle should be used.
      bundle: A generator_pb2.GeneratorBundle object that contains both a
          checkpoint and a metagraph. Or None if a checkpoint should be used.

    Raises:
      SequenceGeneratorException: if neither checkpoint nor bundle is set.
    """
    self._details = details
    self._checkpoint = checkpoint
    self._bundle = bundle

    if self._checkpoint is None and self._bundle is None:
      raise SequenceGeneratorException(
          'Either checkpoint or bundle must be set')
    if self._checkpoint is not None and self._bundle is not None:
      raise SequenceGeneratorException(
          'Checkpoint and bundle cannot both be set')

    if self._bundle:
      if self._bundle.generator_details.id != self._details.id:
        raise SequenceGeneratorException(
            'Generator id in bundle (%s) does not match this generator\'s id '
            '(%s)' % (self._bundle.generator_details.id, self._details.id))

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
  def _initialize_with_checkpoint(self, checkpoint_file):
    """Implementation for building the TF graph given a checkpoint file.

    Args:
      checkpoint_file: The path to the checkpoint file that should be used.
    """
    pass

  @abc.abstractmethod
  def _initialize_with_checkpoint_and_metagraph(self, checkpoint_file,
                                                metagraph_file):
    """Implementation for building the TF graph with a checkpoint and metagraph.

    The implementation should not expect the checkpoint_file and metagraph_file
    to be available after the method returns.

    Args:
      checkpoint_file: The path to the checkpoint file that should be used.
      metagraph_file: The path to the metagraph file that should be used.
    """
    pass

  @abc.abstractmethod
  def _close(self):
    """Implementation for closing the TF session."""
    pass

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

  @abc.abstractmethod
  def _write_checkpoint_with_metagraph(self, checkpoint_filename):
    """Implementation for writing the checkpoint and metagraph.

    Saver should be initialized with sharded=False, and save should be called
    with: meta_graph_suffix='meta', write_meta_graph=True.

    Args:
      checkpoint_filename: Path to the checkpoint file. Should be passed as the
          save_path argument to Saver.save.
    """
    pass

  def initialize(self):
    """Builds the TF graph and loads the checkpoint.

    If the graph has already been initialized, this is a no-op.

    Raises:
      SequenceGeneratorException: If the checkpoint cannot be found.
    """
    if self._initialized:
      return

    # Either self._checkpoint or self._bundle should be set.
    # This is enforced by the constructor.
    if self._checkpoint is not None:
      if not tf.gfile.Exists(self._checkpoint):
        raise SequenceGeneratorException(
            'Checkpoint path does not exist: %s' % (self._checkpoint))
      checkpoint_file = self._checkpoint
      # If this is a directory, try to determine the latest checkpoint in it.
      if tf.gfile.IsDirectory(checkpoint_file):
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_file)
      if checkpoint_file is None:
        raise SequenceGeneratorException(
            'No checkpoint file found in directory: %s' % self._checkpoint)
      if (not tf.gfile.Exists(checkpoint_file) or
          tf.gfile.IsDirectory(checkpoint_file)):
        raise SequenceGeneratorException(
            'Checkpoint path is not a file: %s (supplied path: %s)' % (
                checkpoint_file, self._checkpoint))
      self._initialize_with_checkpoint(checkpoint_file)
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

        self._initialize_with_checkpoint_and_metagraph(
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
      self._close()
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

  def create_bundle_file(self, bundle_file, description=None):
    """Writes a generator_pb2.GeneratorBundle file in the specified location.

    Saves the checkpoint, metagraph, and generator id in one file.

    Args:
      bundle_file: Location to write the bundle file.
      description: A short, human-readable text description of the bundle (e.g.,
          training data, hyper parameters, etc.).

    Raises:
      SequenceGeneratorException: if there is an error creating the bundle file.
    """
    if not bundle_file:
      raise SequenceGeneratorException('Bundle file location not specified.')

    self.initialize()

    tempdir = None
    try:
      tempdir = tempfile.mkdtemp()
      checkpoint_filename = os.path.join(tempdir, 'model.ckpt')

      self._write_checkpoint_with_metagraph(checkpoint_filename)

      if not os.path.isfile(checkpoint_filename):
        raise SequenceGeneratorException(
            'Could not read checkpoint file: %s' % (checkpoint_filename))
      metagraph_filename = checkpoint_filename + '.meta'
      if not os.path.isfile(metagraph_filename):
        raise SequenceGeneratorException(
            'Could not read metagraph file: %s' % (metagraph_filename))

      bundle = generator_pb2.GeneratorBundle()
      bundle.generator_details.CopyFrom(self.details)
      if description is not None:
        bundle.bundle_details.description = description
      with tf.gfile.Open(checkpoint_filename, 'rb') as f:
        bundle.checkpoint_file.append(f.read())
      with tf.gfile.Open(metagraph_filename, 'rb') as f:
        bundle.metagraph_file = f.read()

      with tf.gfile.Open(bundle_file, 'wb') as f:
        f.write(bundle.SerializeToString())
    finally:
      if tempdir is not None:
        tf.gfile.DeleteRecursively(tempdir)
