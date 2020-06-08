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

"""Tests for melspec_input TF/tflite input feature library."""

import tempfile

from magenta.models.onsets_frames_transcription import melspec_input
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.lite.python import convert  # pylint: disable=g-direct-tensorflow-import

tf.disable_v2_behavior()


def _TmpFilePath(suffix):
  """Returns the path to a new temporary file."""
  f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
  return f.name


class MelspecInputTest(tf.test.TestCase):

  # We make this a class method so we can keep it close to Verify...
  def MakeTestWaveform(self):  # self is unused.
    """Generate a 1 sec sweep waveform as a test input."""
    sample_rate = 16000.0
    duration = 0.975  # Exactly 96 frames at 25 ms / 10 ms.
    start_freq = 400.0
    end_freq = 1600.0

    times = np.arange(0, duration, 1 / sample_rate)
    # Exponential frequency sweep.
    frequencies = start_freq * np.exp(
        times / duration * np.log(end_freq / start_freq))
    delta_phases = frequencies * 1 / sample_rate * 2 * np.pi
    phases = np.cumsum(delta_phases)
    # Raised cosine envelope.
    envelope = 0.5 * (1.0 - np.cos(2 * np.pi * times / duration))
    # Final test waveform.
    return envelope * np.cos(phases)

  # We make this a class method so it can use the TestCase assert methods.
  def VerifyMelSpectrumPatch(self, features):
    """Perform tests on melspectrum as calculated for test waveform."""
    expected_time_steps = 96
    expected_mel_bands = 64
    self.assertEqual((expected_time_steps, expected_mel_bands), features.shape)
    # Expect peak magnitude to be somewhere near expected_time_steps/2 == 48
    # (due to raised cosine envelope.)  It isn't *exactly* in the middle because
    # the interaction between the sweeping tone and the mel bands causes some
    # ripple atop the columnwise max.  The peok is actually at frame 43.
    peak_frame = np.argmax(np.max(features, axis=1))
    self.assertGreater(peak_frame, 40)
    self.assertLess(peak_frame, 56)
    # Expect peak frequencies to move up with sweep.  These are "golden",
    # but agree with predictions from instantaneous frequencies and mel scale.
    self.assertEqual(np.argmax(features[20]), 11)
    self.assertEqual(np.argmax(features[42]), 15)
    self.assertEqual(np.argmax(features[64]), 20)

  def BuildTfGraph(self, tflite_compatible=False):
    """Setup the TF graph using the single function under test."""
    if tflite_compatible:
      # tflite requires explicit input sizing.
      input_length = len(self._test_waveform)
    else:
      input_length = None
    with self._graph.as_default():
      waveform_input = tf.placeholder(tf.float32, [input_length])
      # This is the single function provided by the library.
      features = melspec_input.build_mel_calculation_graph(
          waveform_input, tflite_compatible=tflite_compatible)
    self._input = waveform_input
    self._output = features

  def RunTfGraph(self):
    """Return output of running the current graph under TF."""
    feature_output = self._session.run(
        self._output, feed_dict={self._input: self._test_waveform})
    return feature_output

  def BuildAndRunTfGraph(self, tflite_compatible=False):
    """Build the graph then run it."""
    self.BuildTfGraph(tflite_compatible)
    return self.RunTfGraph()

  def setUp(self):
    self._test_waveform = self.MakeTestWaveform()
    # Initialize TensorFlow.
    self._graph = tf.Graph()
    self._session = tf.Session(graph=self._graph)

  def testPlainTfFeatureCalculation(self):
    """Test simple TF feature calculation."""
    feature_output = self.BuildAndRunTfGraph(tflite_compatible=False)
    # Only one patch returned.
    self.assertEqual(1, feature_output.shape[0])
    self.VerifyMelSpectrumPatch(feature_output[0])

  def testTfLiteGraphAgainstPlainTf(self):
    """Test the tflite graph running under plain TF."""
    plain_tf_output = self.BuildAndRunTfGraph(tflite_compatible=False)
    tflite_output = self.BuildAndRunTfGraph(tflite_compatible=True)
    # Results don't match to 6 decimals, 1 is OK.
    # TODO(fjord): Eventually switch to implementation that has fewer
    # differences.
    np.testing.assert_allclose(
        tflite_output[0], plain_tf_output[0], rtol=.05, atol=.3)

  def RunTfliteCompiler(self):
    # Attempt to run the tflite-style conversion to the current graph.
    converter = tf.lite.TFLiteConverter.from_session(
        self._session, [self._input], [self._output])
    converter.inference_type = tf.lite.constants.FLOAT
    tflite_model = converter.convert()
    output_filename = _TmpFilePath(suffix='.tflite')
    open(output_filename, 'wb').write(tflite_model)
    return output_filename

  def testTfLiteCompiles(self):
    """Check that we can compile the tflite graph (i.e., no invalid ops)."""
    self.BuildTfGraph(tflite_compatible=True)
    self.RunTfliteCompiler()

  def testRegularTfGraphIsntTfLiteCompatible(self):
    self.BuildTfGraph(tflite_compatible=False)
    with self.assertRaises(convert.ConverterError):
      self.RunTfliteCompiler()

  def RunTfliteModel(self, tflite_model_path):
    """Load and run TFLite model under the interpreter."""
    interpreter = tf.lite.Interpreter(tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'],
                           np.array(self._test_waveform, dtype=np.float32))
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

  def testTfLiteGraphUnderTfLite(self):
    """Verify output of tflite interpreter matches plain TF output."""
    self.BuildTfGraph(tflite_compatible=True)
    tf_output = self.RunTfGraph()
    # Graph is now built in the current session, ready for tflite conversion.
    tflite_filename = self.RunTfliteCompiler()
    # Run the tflite model with the tflite interpreter.
    tflite_output = self.RunTfliteModel(tflite_filename)
    # Be satisfied with 1 d.p. (i.e., 2 sf) agreement.
    # At 2 d.p., we got 0.07% disagreement, probably just 1 value.)
    np.testing.assert_array_almost_equal(
        tflite_output[0], tf_output[0], decimal=1)


if __name__ == '__main__':
  tf.test.main()
