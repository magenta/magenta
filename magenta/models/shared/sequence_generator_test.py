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

"""Tests for sequence_generator."""

from magenta.models.shared import model
from magenta.models.shared import sequence_generator
from note_seq.protobuf import generator_pb2
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class Model(model.BaseModel):
  """Test model."""

  def _build_graph_for_generation(self):
    pass


class SeuenceGenerator(sequence_generator.BaseSequenceGenerator):
  """Test generator."""

  def __init__(self, checkpoint=None, bundle=None):
    details = generator_pb2.GeneratorDetails(
        id='test_generator',
        description='Test Generator')

    super(SeuenceGenerator, self).__init__(
        Model(), details, checkpoint=checkpoint,
        bundle=bundle)

  def _generate(self):
    pass


class SequenceGeneratorTest(tf.test.TestCase):

  def testSpecifyEitherCheckPointOrBundle(self):
    bundle = generator_pb2.GeneratorBundle(
        generator_details=generator_pb2.GeneratorDetails(
            id='test_generator'),
        checkpoint_file=[b'foo.ckpt'],
        metagraph_file=b'foo.ckpt.meta')

    with self.assertRaises(sequence_generator.SequenceGeneratorError):
      SeuenceGenerator(checkpoint='foo.ckpt', bundle=bundle)
    with self.assertRaises(sequence_generator.SequenceGeneratorError):
      SeuenceGenerator(checkpoint=None, bundle=None)

    SeuenceGenerator(checkpoint='foo.ckpt')
    SeuenceGenerator(bundle=bundle)

  def testUseMatchingGeneratorId(self):
    bundle = generator_pb2.GeneratorBundle(
        generator_details=generator_pb2.GeneratorDetails(
            id='test_generator'),
        checkpoint_file=[b'foo.ckpt'],
        metagraph_file=b'foo.ckpt.meta')

    SeuenceGenerator(bundle=bundle)

    bundle.generator_details.id = 'blarg'

    with self.assertRaises(sequence_generator.SequenceGeneratorError):
      SeuenceGenerator(bundle=bundle)

  def testGetBundleDetails(self):
    # Test with non-bundle generator.
    seq_gen = SeuenceGenerator(checkpoint='foo.ckpt')
    self.assertIsNone(seq_gen.bundle_details)

    # Test with bundle-based generator.
    bundle_details = generator_pb2.GeneratorBundle.BundleDetails(
        description='bundle of joy')
    bundle = generator_pb2.GeneratorBundle(
        generator_details=generator_pb2.GeneratorDetails(
            id='test_generator'),
        bundle_details=bundle_details,
        checkpoint_file=[b'foo.ckpt'],
        metagraph_file=b'foo.ckpt.meta')
    seq_gen = SeuenceGenerator(bundle=bundle)
    self.assertEqual(bundle_details, seq_gen.bundle_details)


if __name__ == '__main__':
  tf.test.main()
