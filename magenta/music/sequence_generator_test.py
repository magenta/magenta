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
"""Tests for sequence_generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports

import tensorflow as tf

from magenta.music import model
from magenta.music import sequence_generator
from magenta.protobuf import generator_pb2


class TestModel(model.BaseModel):

  def __init__(self):
    super(TestModel, self).__init__()

  def _build_graph_for_generation(self):
    pass


class TestSequenceGenerator(sequence_generator.BaseSequenceGenerator):

  def __init__(self, checkpoint=None, bundle=None):
    details = generator_pb2.GeneratorDetails(
        id='test_generator',
        description='Test Generator')

    super(TestSequenceGenerator, self).__init__(
        TestModel(), details, checkpoint=checkpoint,
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

    with self.assertRaises(sequence_generator.SequenceGeneratorException):
      TestSequenceGenerator(checkpoint='foo.ckpt', bundle=bundle)
    with self.assertRaises(sequence_generator.SequenceGeneratorException):
      TestSequenceGenerator(checkpoint=None, bundle=None)

    TestSequenceGenerator(checkpoint='foo.ckpt')
    TestSequenceGenerator(bundle=bundle)

  def testUseMatchingGeneratorId(self):
    bundle = generator_pb2.GeneratorBundle(
        generator_details=generator_pb2.GeneratorDetails(
            id='test_generator'),
        checkpoint_file=[b'foo.ckpt'],
        metagraph_file=b'foo.ckpt.meta')

    TestSequenceGenerator(bundle=bundle)

    bundle.generator_details.id = 'blarg'

    with self.assertRaises(sequence_generator.SequenceGeneratorException):
      TestSequenceGenerator(bundle=bundle)

  def testGetBundleDetails(self):
    # Test with non-bundle generator.
    seq_gen = TestSequenceGenerator(checkpoint='foo.ckpt')
    self.assertEquals(None, seq_gen.bundle_details)

    # Test with bundle-based generator.
    bundle_details = generator_pb2.GeneratorBundle.BundleDetails(
        description='bundle of joy')
    bundle = generator_pb2.GeneratorBundle(
        generator_details=generator_pb2.GeneratorDetails(
            id='test_generator'),
        bundle_details=bundle_details,
        checkpoint_file=[b'foo.ckpt'],
        metagraph_file=b'foo.ckpt.meta')
    seq_gen = TestSequenceGenerator(bundle=bundle)
    self.assertEquals(bundle_details, seq_gen.bundle_details)


if __name__ == '__main__':
  tf.test.main()
