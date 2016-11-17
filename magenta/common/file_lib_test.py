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
"""Tests for tf_lib."""

# internal imports
import tensorflow as tf

from magenta.common import file_lib


class FileLibtest(tf.test.TestCase):

  def testIsShardedFilename(self):
    self.assertTrue(file_lib.is_sharded_filename('/tmp/sequence_examples_1@3'))
    self.assertTrue(file_lib.is_sharded_filename('/tmp/examples.tf@5'))
    self.assertFalse(file_lib.is_sharded_filename('/tmp/sequence_examples_1@N'))
    self.assertFalse(file_lib.is_sharded_filename('/tmp/sequence_examples_1'))

  def testShardedFilenameToList(self):
    self.assertEquals(
        file_lib.sharded_filename_to_list('/tmp/examples@2'),
        ['/tmp/examples-00000-of-00002', '/tmp/examples-00001-of-00002'])
    with self.assertRaises(ValueError):
      file_lib.sharded_filename_to_list('/tmp/examples')
    with self.assertRaises(ValueError):
      file_lib.sharded_filename_to_list('/tmp/examples@0')

  def testGetFilenameList(self):
    self.assertEquals(
        file_lib.get_filename_list('/tmp/examples@2'),
        ['/tmp/examples-00000-of-00002', '/tmp/examples-00001-of-00002'])
    self.assertEquals(
        file_lib.get_filename_list('/tmp/examples'), ['/tmp/examples'])


if __name__ == '__main__':
  tf.test.main()
