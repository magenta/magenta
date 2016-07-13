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


class PipelineTest(tf.test.TestCase):

  def setUp(self):
    pass

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
    class MockProto(object):

      def __init__(self, string=''):
        self.string = string

      @staticmethod
      def FromString(string):
        return MockProto(string)

      def __eq__(self, other):
        return isinstance(other, MockProto) and self.string == other.string

    tfrecord_file = os.path.join(
        tf.resource_loader.get_data_files_path(),
        '../testdata/tfrecord_iterator_test.tfrecord')
    self.assertEqual(
        [MockProto(string) for string in ['hello world', '12345', 'success']],
        list(pipeline.tf_record_iterator(tfrecord_file, MockProto)))

  def testRunPipelineSerial(self):
    pass

  def testPipelineIterator(self):
    pass


if __name__ == '__main__':
  tf.test.main()