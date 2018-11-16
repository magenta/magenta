# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Tests for performance controls."""

import tensorflow as tf

from magenta.music import performance_controls
from magenta.music import performance_lib


class NoteDensityPerformanceControlSignalTest(tf.test.TestCase):

  def setUp(self):
    self.control = performance_controls.NoteDensityPerformanceControlSignal(
        window_size_seconds=1.0, density_bin_ranges=[1.0, 5.0])

  def testExtract(self):
    performance = performance_lib.Performance(steps_per_second=100)

    pe = performance_lib.PerformanceEvent
    perf_events = [
        pe(pe.NOTE_ON, 60),
        pe(pe.NOTE_ON, 64),
        pe(pe.NOTE_ON, 67),
        pe(pe.TIME_SHIFT, 50),
        pe(pe.NOTE_OFF, 60),
        pe(pe.NOTE_OFF, 64),
        pe(pe.TIME_SHIFT, 25),
        pe(pe.NOTE_OFF, 67),
        pe(pe.NOTE_ON, 64),
        pe(pe.TIME_SHIFT, 25),
        pe(pe.NOTE_OFF, 64)
    ]
    for event in perf_events:
      performance.append(event)

    expected_density_sequence = [
        4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 0.0]

    density_sequence = self.control.extract(performance)
    self.assertEqual(expected_density_sequence, density_sequence)

  def testEncoder(self):
    density_sequence = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

    expected_inputs = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]

    self.assertEqual(expected_inputs[0],
                     self.control.encoder.events_to_input(density_sequence, 0))
    self.assertEqual(expected_inputs[1],
                     self.control.encoder.events_to_input(density_sequence, 1))
    self.assertEqual(expected_inputs[2],
                     self.control.encoder.events_to_input(density_sequence, 2))
    self.assertEqual(expected_inputs[3],
                     self.control.encoder.events_to_input(density_sequence, 3))
    self.assertEqual(expected_inputs[4],
                     self.control.encoder.events_to_input(density_sequence, 4))
    self.assertEqual(expected_inputs[5],
                     self.control.encoder.events_to_input(density_sequence, 5))


class PitchHistogramPerformanceControlSignalTest(tf.test.TestCase):

  def setUp(self):
    self.control = performance_controls.PitchHistogramPerformanceControlSignal(
        window_size_seconds=1.0, prior_count=0)

  def testExtract(self):
    performance = performance_lib.Performance(steps_per_second=100)

    pe = performance_lib.PerformanceEvent
    perf_events = [
        pe(pe.NOTE_ON, 60),
        pe(pe.NOTE_ON, 64),
        pe(pe.NOTE_ON, 67),
        pe(pe.TIME_SHIFT, 50),
        pe(pe.NOTE_OFF, 60),
        pe(pe.NOTE_OFF, 64),
        pe(pe.TIME_SHIFT, 25),
        pe(pe.NOTE_OFF, 67),
        pe(pe.NOTE_ON, 64),
        pe(pe.TIME_SHIFT, 25),
        pe(pe.NOTE_OFF, 64)
    ]
    for event in perf_events:
      performance.append(event)

    expected_histogram_sequence = [
        [0.5, 0, 0, 0, 0.75, 0, 0, 0.75, 0, 0, 0, 0],
        [0.5, 0, 0, 0, 0.75, 0, 0, 0.75, 0, 0, 0, 0],
        [0.5, 0, 0, 0, 0.75, 0, 0, 0.75, 0, 0, 0, 0],
        [0.5, 0, 0, 0, 0.75, 0, 0, 0.75, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.25, 0, 0, 0.25, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.25, 0, 0, 0.25, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.25, 0, 0, 0.25, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.25, 0, 0, 0.0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.25, 0, 0, 0.0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.25, 0, 0, 0.0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    histogram_sequence = self.control.extract(performance)
    self.assertEqual(expected_histogram_sequence, histogram_sequence)

  def testEncoder(self):
    histogram_sequence = [
        [0.5, 0, 0, 0, 0.75, 0, 0, 0.75, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.25, 0, 0, 0.25, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.25, 0, 0, 0.0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    expected_inputs = [
        [0.25, 0, 0, 0, 0.375, 0, 0, 0.375, 0, 0, 0, 0],
        [0.0, 0, 0, 0, 0.5, 0, 0, 0.5, 0, 0, 0, 0],
        [0.0, 0, 0, 0, 1.0, 0, 0, 0.0, 0, 0, 0, 0],
        [1.0 / 12.0] * 12
    ]

    self.assertEqual(
        expected_inputs[0],
        self.control.encoder.events_to_input(histogram_sequence, 0))
    self.assertEqual(
        expected_inputs[1],
        self.control.encoder.events_to_input(histogram_sequence, 1))
    self.assertEqual(
        expected_inputs[2],
        self.control.encoder.events_to_input(histogram_sequence, 2))
    self.assertEqual(
        expected_inputs[3],
        self.control.encoder.events_to_input(histogram_sequence, 3))


if __name__ == '__main__':
  tf.test.main()
