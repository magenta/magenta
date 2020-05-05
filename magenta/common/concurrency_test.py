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

"""Tests for concurrency."""

import threading
import time

from magenta.common import concurrency
import tensorflow.compat.v1 as tf


class ConcurrencyTest(tf.test.TestCase):

  def testSleeper_SleepUntil(self):
    # Burn in.
    for _ in range(10):
      concurrency.Sleeper().sleep(.01)

    future_time = time.time() + 0.5
    concurrency.Sleeper().sleep_until(future_time)
    self.assertAlmostEqual(time.time(), future_time, delta=0.005)

  def testSleeper_Sleep(self):
    # Burn in.
    for _ in range(10):
      concurrency.Sleeper().sleep(.01)

    def sleep_test_thread(duration):
      start_time = time.time()
      concurrency.Sleeper().sleep(duration)
      self.assertAlmostEqual(time.time(), start_time + duration, delta=0.005)

    threads = [threading.Thread(target=sleep_test_thread, args=[i * 0.1])
               for i in range(10)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()


if __name__ == '__main__':
  tf.test.main()
