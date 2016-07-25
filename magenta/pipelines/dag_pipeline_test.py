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
"""Tests for dag_pipeline."""


# internal imports
import tensorflow as tf

from magenta.pipelines import dag_pipeline
from magenta.pipelines import pipeline
from magenta.pipelines import statistics


class DAGPipelineTest(tf.test.TestCase):

  def testInputAndOutputTypes(self):
    # Tests single object and dictionaries for input_type and output_type.
    pass

  def testMultiOutput(self):
    # Tests a Pipeline that maps a single input to multiple outputs.
    pass

  def testUnequalOutputCounts(self):
    # Tests dictionary output type where each output list has a different size.
    pass

  def testIntermediateUnequalOutputCounts(self):
    # Tests that intermediate output lists which are not the same length are handled correctly.
    pass

  def testDictionaryToDictionaryConnection(self):
    # Tests a direct dict to dict connection in the DAG.
    pass

  def testInput(self):
    # Tests Input object.
    pass

  def testOutput(self):
    # Tests Output object.
    pass

  def testStatistics(self):
    pass

  def testGraphCycleException(self):
    pass

  def testInputTypeDoesntMatchOutputTypeException(self):
    pass

  def testActualOutputTypeDoesntMatchGivenOutputTypeException(self):
    pass

  def testPipelineKeyError(self):
    # This happens if Key() is used on a pipeline with out a dictionary output, or the key is not in the output_type dict.
    pass

  def testInputDictDoesntMatchOutputDictException(self):
    # This happens in a direct connection where input_type and output_type do not match.
    pass

  def testInvalidStatisticException(self):
    pass


