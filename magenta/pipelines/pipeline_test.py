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
"""Tests for pipeline."""

import os
import tempfile

# internal imports
import tensorflow as tf

from magenta.pipelines import pipeline


class MockProto(object):

  def __init__(self, string=''):
    self.string = string

  @staticmethod
  def FromString(string):
    return MockProto(string)

  def SerializeToString(self):
    return 'serialized:' + self.string

  def __eq__(self, other):
    return isinstance(other, MockProto) and self.string == other.string

  def __hash__(self):
    return hash(self.string)


class MockPipeline(pipeline.Pipeline):

  def transform(self, input_object):
    return {
        'dataset_1': [
            MockProto(input_object + '_A'),
            MockProto(input_object + '_B')],
        'dataset_2': [MockProto(input_object + '_C')]}

  def get_output_names(self):
    return ['dataset_1', 'dataset_2']


class PipelineTest(tf.test.TestCase):

  def testRecursiveFileIterator(self):
    target_files = [
        ('0.ext', 'hello world'),
        ('a/1.ext', '123456'),
        ('a/2.ext', 'abcd'),
        ('b/c/3.ext', '9999'),
        ('b/z/3.ext', 'qwerty'),
        ('d/4.ext', 'mary had a little lamb'),
        ('d/e/5.ext', 'zzzzzzzz'),
        ('d/e/f/g/6.ext', 'yyyyyyyyyyy')]
    extra_files = [
        ('stuff.txt', 'some stuff'),
        ('a/q/r/file', 'more stuff')]

    root_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    for path, contents in target_files + extra_files:
      abs_path = os.path.join(root_dir, path)
      tf.gfile.MakeDirs(os.path.dirname(abs_path))
      tf.gfile.FastGFile(abs_path, mode='w').write(contents)

    file_iterator = pipeline.recursive_file_iterator(root_dir, 'ext')

    self.assertEqual(set([contents for _, contents in target_files]),
                     set(file_iterator))

  def testTFRecordIterator(self):
    tfrecord_file = os.path.join(
        tf.resource_loader.get_data_files_path(),
        '../testdata/tfrecord_iterator_test.tfrecord')
    self.assertEqual(
        [MockProto(string) for string in ['hello world', '12345', 'success']],
        list(pipeline.tf_record_iterator(tfrecord_file, MockProto)))

  def testRunPipelineSerial(self):
    strings = ['abcdefg', 'helloworld!', 'qwerty']
    root_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    pipeline.run_pipeline_serial(
        MockPipeline(), iter(strings), root_dir)

    dataset_1_dir = os.path.join(root_dir, 'dataset_1.tfrecord')
    dataset_2_dir = os.path.join(root_dir, 'dataset_2.tfrecord')
    self.assertTrue(tf.gfile.Exists(dataset_1_dir))
    self.assertTrue(tf.gfile.Exists(dataset_2_dir))

    dataset_1_reader = tf.python_io.tf_record_iterator(dataset_1_dir)
    self.assertEqual(
        set(['serialized:%s_A' % s for s in strings] +
            ['serialized:%s_B' % s for s in strings]),
        set(dataset_1_reader))

    dataset_2_reader = tf.python_io.tf_record_iterator(dataset_2_dir)
    self.assertEqual(
        set(['serialized:%s_C' % s for s in strings]),
        set(dataset_2_reader))

  def testPipelineIterator(self):
    strings = ['abcdefg', 'helloworld!', 'qwerty']
    result = pipeline.load_pipeline(MockPipeline(), iter(strings))

    self.assertEqual(
        set([MockProto(s + '_A') for s in strings] +
            [MockProto(s + '_B') for s in strings]),
        set(result['dataset_1']))
    self.assertEqual(
        set([MockProto(s + '_C') for s in strings]),
        set(result['dataset_2']))


if __name__ == '__main__':
  tf.test.main()
