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

"""Tests for dag_pipeline."""

import collections

from absl.testing import absltest
from magenta.pipelines import dag_pipeline
from magenta.pipelines import pipeline
from magenta.pipelines import statistics

Type0 = collections.namedtuple('Type0', ['x', 'y', 'z'])
Type1 = collections.namedtuple('Type1', ['x', 'y'])
Type2 = collections.namedtuple('Type2', ['z'])
Type3 = collections.namedtuple('Type3', ['s', 't'])
Type4 = collections.namedtuple('Type4', ['s', 't', 'z'])
Type5 = collections.namedtuple('Type5', ['a', 'b', 'c', 'd', 'z'])

# pylint:disable=missing-class-docstring


class UnitA(pipeline.Pipeline):

  def __init__(self):
    pipeline.Pipeline.__init__(self, Type0, {'t1': Type1, 't2': Type2})

  def transform(self, input_object):
    t1 = Type1(x=input_object.x, y=input_object.y)
    t2 = Type2(z=input_object.z)
    return {'t1': [t1], 't2': [t2]}


class UnitB(pipeline.Pipeline):

  def __init__(self):
    pipeline.Pipeline.__init__(self, Type1, Type3)

  def transform(self, input_object):
    t3 = Type3(s=input_object.x * 1000, t=input_object.y - 100)
    return [t3]


class UnitC(pipeline.Pipeline):

  def __init__(self):
    pipeline.Pipeline.__init__(
        self,
        {'A_data': Type2, 'B_data': Type3},
        {'regular_data': Type4, 'special_data': Type4})

  def transform(self, input_object):
    s = input_object['B_data'].s
    t = input_object['B_data'].t
    z = input_object['A_data'].z
    regular = Type4(s=s, t=t, z=0)
    special = Type4(s=s + z * 100, t=t - z * 100, z=z)
    return {'regular_data': [regular], 'special_data': [special]}


class UnitD(pipeline.Pipeline):

  def __init__(self):
    pipeline.Pipeline.__init__(
        self, {'0': Type4, '1': Type3, '2': Type4}, Type5)

  def transform(self, input_object):
    assert input_object['1'].s == input_object['0'].s
    assert input_object['1'].t == input_object['0'].t
    t5 = Type5(
        a=input_object['0'].s, b=input_object['0'].t,
        c=input_object['2'].s, d=input_object['2'].t, z=input_object['2'].z)
    return [t5]


class DAGPipelineTest(absltest.TestCase):

  def testDAGPipelineInputAndOutputType(self):
    # Tests that the DAGPipeline has the correct `input_type` and
    # `output_type` values based on the DAG given to it.
    a, b, c, d = UnitA(), UnitB(), UnitC(), UnitD()

    dag = {a: dag_pipeline.DagInput(Type0),
           b: a['t1'],
           c: {'A_data': a['t2'], 'B_data': b},
           d: {'0': c['regular_data'], '1': b, '2': c['special_data']},
           dag_pipeline.DagOutput('abcdz'): d}
    dag_pipe_obj = dag_pipeline.DAGPipeline(dag)
    self.assertEqual(dag_pipe_obj.input_type, Type0)
    self.assertEqual(dag_pipe_obj.output_type, {'abcdz': Type5})

    dag = {a: dag_pipeline.DagInput(Type0),
           dag_pipeline.DagOutput('t1'): a['t1'],
           dag_pipeline.DagOutput('t2'): a['t2']}
    dag_pipe_obj = dag_pipeline.DAGPipeline(dag)
    self.assertEqual(dag_pipe_obj.input_type, Type0)
    self.assertEqual(dag_pipe_obj.output_type, {'t1': Type1, 't2': Type2})

  def testSingleOutputs(self):
    # Tests single object and dictionaries in the DAG.
    a, b, c, d = UnitA(), UnitB(), UnitC(), UnitD()
    dag = {a: dag_pipeline.DagInput(Type0),
           b: a['t1'],
           c: {'A_data': a['t2'], 'B_data': b},
           d: {'0': c['regular_data'], '1': b, '2': c['special_data']},
           dag_pipeline.DagOutput('abcdz'): d}

    dag_pipe_obj = dag_pipeline.DAGPipeline(dag)
    inputs = [Type0(1, 2, 3), Type0(-1, -2, -3), Type0(3, -3, 2)]

    for input_object in inputs:
      x, y, z = input_object.x, input_object.y, input_object.z
      output_dict = dag_pipe_obj.transform(input_object)

      self.assertEqual(list(output_dict.keys()), ['abcdz'])
      results = output_dict['abcdz']
      self.assertLen(results, 1)
      result = results[0]

      # The following outputs are the result of passing the values in
      # `input_object` through the transform functions of UnitA, UnitB, UnitC,
      # and UnitD (all defined at the top of this file), connected in the way
      # defined by `dag`.
      self.assertEqual(result.a, x * 1000)
      self.assertEqual(result.b, y - 100)
      self.assertEqual(result.c, x * 1000 + z * 100)
      self.assertEqual(result.d, y - 100 - z * 100)
      self.assertEqual(result.z, z)

  def testMultiOutput(self):
    # Tests a pipeline.Pipeline that maps a single input to multiple outputs.

    class UnitQ(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, {'t1': Type1, 't2': Type2})

      def transform(self, input_object):
        t1 = [Type1(x=input_object.x + i, y=input_object.y + i)
              for i in range(input_object.z)]
        t2 = [Type2(z=input_object.z)]
        return {'t1': t1, 't2': t2}

    q, b, c = UnitQ(), UnitB(), UnitC()
    dag = {q: dag_pipeline.DagInput(Type0),
           b: q['t1'],
           c: {'A_data': q['t2'], 'B_data': b},
           dag_pipeline.DagOutput('outputs'): c['regular_data']}

    dag_pipe_obj = dag_pipeline.DAGPipeline(dag)

    x, y, z = 1, 2, 3
    output_dict = dag_pipe_obj.transform(Type0(x, y, z))

    self.assertEqual(list(output_dict.keys()), ['outputs'])
    results = output_dict['outputs']
    self.assertLen(results, 3)

    expected_results = [Type4((x + i) * 1000, (y + i) - 100, 0)
                        for i in range(z)]
    self.assertEqual(set(results), set(expected_results))

  def testUnequalOutputCounts(self):
    # Tests dictionary output type where each output list has a different size.

    class UnitQ(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, Type1)

      def transform(self, input_object):
        return [Type1(x=input_object.x + i, y=input_object.y + i)
                for i in range(input_object.z)]

    class Partitioner(pipeline.Pipeline):

      def __init__(self, input_type, training_set_name, test_set_name):
        self.training_set_name = training_set_name
        self.test_set_name = test_set_name
        pipeline.Pipeline.__init__(
            self,
            input_type,
            {training_set_name: input_type, test_set_name: input_type})

      def transform(self, input_object):
        if input_object.x < 0:
          return {self.training_set_name: [],
                  self.test_set_name: [input_object]}
        return {self.training_set_name: [input_object], self.test_set_name: []}

    q = UnitQ()
    partition = Partitioner(q.output_type, 'training_set', 'test_set')

    dag = {q: dag_pipeline.DagInput(q.input_type),
           partition: q,
           dag_pipeline.DagOutput('training_set'): partition['training_set'],
           dag_pipeline.DagOutput('test_set'): partition['test_set']}

    dag_pipe_obj = dag_pipeline.DAGPipeline(dag)
    x, y, z = -3, 0, 8
    output_dict = dag_pipe_obj.transform(Type0(x, y, z))

    self.assertEqual(set(output_dict.keys()), set(['training_set', 'test_set']))
    training_results = output_dict['training_set']
    test_results = output_dict['test_set']

    expected_training_results = [Type1(x + i, y + i) for i in range(-x, z)]
    expected_test_results = [Type1(x + i, y + i) for i in range(0, -x)]
    self.assertEqual(set(training_results), set(expected_training_results))
    self.assertEqual(set(test_results), set(expected_test_results))

  def testIntermediateUnequalOutputCounts(self):
    # Tests that intermediate output lists which are not the same length are
    # handled correctly.

    class UnitQ(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, {'xy': Type1, 'z': Type2})

      def transform(self, input_object):
        return {'xy': [Type1(x=input_object.x + i, y=input_object.y + i)
                       for i in range(input_object.z)],
                'z': [Type2(z=i) for i in [-input_object.z, input_object.z]]}

    class Partitioner(pipeline.Pipeline):

      def __init__(self, input_type, training_set_name, test_set_name):
        self.training_set_name = training_set_name
        self.test_set_name = test_set_name
        pipeline.Pipeline.__init__(
            self,
            input_type,
            {training_set_name: Type0, test_set_name: Type0})

      def transform(self, input_dict):
        input_object = Type0(input_dict['xy'].x,
                             input_dict['xy'].y,
                             input_dict['z'].z)
        if input_object.x < 0:
          return {self.training_set_name: [],
                  self.test_set_name: [input_object]}
        return {self.training_set_name: [input_object], self.test_set_name: []}

    q = UnitQ()
    partition = Partitioner(q.output_type, 'training_set', 'test_set')

    dag = {q: dag_pipeline.DagInput(q.input_type),
           partition: {'xy': q['xy'], 'z': q['z']},
           dag_pipeline.DagOutput('training_set'): partition['training_set'],
           dag_pipeline.DagOutput('test_set'): partition['test_set']}

    dag_pipe_obj = dag_pipeline.DAGPipeline(dag)
    x, y, z = -3, 0, 8
    output_dict = dag_pipe_obj.transform(Type0(x, y, z))

    self.assertEqual(set(output_dict.keys()), set(['training_set', 'test_set']))
    training_results = output_dict['training_set']
    test_results = output_dict['test_set']

    all_expected_results = [Type0(x + i, y + i, zed)
                            for i in range(0, z) for zed in [-z, z]]
    expected_training_results = [sample for sample in all_expected_results
                                 if sample.x >= 0]
    expected_test_results = [sample for sample in all_expected_results
                             if sample.x < 0]
    self.assertEqual(set(training_results), set(expected_training_results))
    self.assertEqual(set(test_results), set(expected_test_results))

  def testDirectConnection(self):
    # Tests a direct dict to dict connection in the DAG.

    class UnitQ(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, {'xy': Type1, 'z': Type2})

      def transform(self, input_object):
        return {'xy': [Type1(x=input_object.x, y=input_object.y)],
                'z': [Type2(z=input_object.z)]}

    class UnitR(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, {'xy': Type1, 'z': Type2}, Type4)

      def transform(self, input_dict):
        return [Type4(input_dict['xy'].x,
                      input_dict['xy'].y,
                      input_dict['z'].z)]

    q, r = UnitQ(), UnitR()
    dag = {q: dag_pipeline.DagInput(q.input_type),
           r: q,
           dag_pipeline.DagOutput('output'): r}

    dag_pipe_obj = dag_pipeline.DAGPipeline(dag)
    x, y, z = -3, 0, 8
    output_dict = dag_pipe_obj.transform(Type0(x, y, z))

    self.assertEqual(output_dict, {'output': [Type4(x, y, z)]})

  def testOutputConnectedToDict(self):

    class UnitQ(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, {'xy': Type1, 'z': Type2})

      def transform(self, input_object):
        return {'xy': [Type1(x=input_object.x, y=input_object.y)],
                'z': [Type2(z=input_object.z)]}

    q = UnitQ()
    dag = {q: dag_pipeline.DagInput(q.input_type),
           dag_pipeline.DagOutput(): q}
    dag_pipe_obj = dag_pipeline.DAGPipeline(dag)
    self.assertEqual(dag_pipe_obj.output_type, {'xy': Type1, 'z': Type2})
    x, y, z = -3, 0, 8
    output_dict = dag_pipe_obj.transform(Type0(x, y, z))
    self.assertEqual(output_dict, {'xy': [Type1(x, y)], 'z': [Type2(z)]})

    dag = {q: dag_pipeline.DagInput(q.input_type),
           dag_pipeline.DagOutput(): {'xy': q['xy'], 'z': q['z']}}
    dag_pipe_obj = dag_pipeline.DAGPipeline(dag)
    self.assertEqual(dag_pipe_obj.output_type, {'xy': Type1, 'z': Type2})
    x, y, z = -3, 0, 8
    output_dict = dag_pipe_obj.transform(Type0(x, y, z))
    self.assertEqual(output_dict, {'xy': [Type1(x, y)], 'z': [Type2(z)]})

  def testNoOutputs(self):
    # Test that empty lists or dicts as intermediate or final outputs don't
    # break anything.

    class UnitQ(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, {'xy': Type1, 'z': Type2})

      def transform(self, input_object):
        return {'xy': [], 'z': []}

    class UnitR(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, {'xy': Type1, 'z': Type2}, Type4)

      def transform(self, input_dict):
        return [Type4(input_dict['xy'].x,
                      input_dict['xy'].y,
                      input_dict['z'].z)]

    class UnitS(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, Type1)

      def transform(self, unused_input_dict):
        return []

    q, r, s = UnitQ(), UnitR(), UnitS()
    dag = {q: dag_pipeline.DagInput(Type0),
           r: q,
           dag_pipeline.DagOutput('output'): r}
    dag_pipe_obj = dag_pipeline.DAGPipeline(dag)
    self.assertEqual(dag_pipe_obj.transform(Type0(1, 2, 3)), {'output': []})

    dag = {q: dag_pipeline.DagInput(Type0),
           s: dag_pipeline.DagInput(Type0),
           r: {'xy': s, 'z': q['z']},
           dag_pipeline.DagOutput('output'): r}
    dag_pipe_obj = dag_pipeline.DAGPipeline(dag)
    self.assertEqual(dag_pipe_obj.transform(Type0(1, 2, 3)), {'output': []})

    dag = {s: dag_pipeline.DagInput(Type0),
           dag_pipeline.DagOutput('output'): s}
    dag_pipe_obj = dag_pipeline.DAGPipeline(dag)
    self.assertEqual(dag_pipe_obj.transform(Type0(1, 2, 3)), {'output': []})

    dag = {q: dag_pipeline.DagInput(Type0),
           dag_pipeline.DagOutput(): q}
    dag_pipe_obj = dag_pipeline.DAGPipeline(dag)
    self.assertEqual(
        dag_pipe_obj.transform(Type0(1, 2, 3)),
        {'xy': [], 'z': []})

  def testNoPipelines(self):
    dag = {dag_pipeline.DagOutput('output'): dag_pipeline.DagInput(Type0)}
    dag_pipe_obj = dag_pipeline.DAGPipeline(dag)
    self.assertEqual(
        dag_pipe_obj.transform(Type0(1, 2, 3)),
        {'output': [Type0(1, 2, 3)]})

  def testStatistics(self):

    class UnitQ(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, Type1)
        self.stats = []

      def transform(self, input_object):
        self._set_stats([statistics.Counter('output_count', input_object.z)])
        return [Type1(x=input_object.x + i, y=input_object.y + i)
                for i in range(input_object.z)]

    class UnitR(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type1, Type1)

      def transform(self, input_object):
        self._set_stats([statistics.Counter('input_count', 1)])
        return [input_object]

    q, r = UnitQ(), UnitR()
    dag = {q: dag_pipeline.DagInput(q.input_type),
           r: q,
           dag_pipeline.DagOutput('output'): r}
    dag_pipe_obj = dag_pipeline.DAGPipeline(dag, 'DAGPipelineName')
    for x, y, z in [(-3, 0, 8), (1, 2, 3), (5, -5, 5)]:
      dag_pipe_obj.transform(Type0(x, y, z))
      stats_1 = dag_pipe_obj.get_stats()
      stats_2 = dag_pipe_obj.get_stats()
      self.assertEqual(stats_1, stats_2)

      for stat in stats_1:
        self.assertIsInstance(stat, statistics.Counter)

      names = sorted([stat.name for stat in stats_1])
      self.assertEqual(
          names,
          (['DAGPipelineName_UnitQ_output_count'] +
           ['DAGPipelineName_UnitR_input_count'] * z))

      for stat in stats_1:
        if stat.name == 'DAGPipelineName_UnitQ_output_count':
          self.assertEqual(stat.count, z)
        else:
          self.assertEqual(stat.count, 1)

  def testInvalidDAGError(self):
    class UnitQ(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, {'a': Type1, 'b': Type2})

      def transform(self, input_object):
        pass

    class UnitR(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type1, Type2)

      def transform(self, input_object):
        pass

    q, r = UnitQ(), UnitR()

    dag = {q: dag_pipeline.DagInput(Type0),
           UnitR: q,
           dag_pipeline.DagOutput('output'): r}
    with self.assertRaises(dag_pipeline.InvalidDAGError):
      dag_pipeline.DAGPipeline(dag)

    dag = {q: dag_pipeline.DagInput(Type0),
           'r': q,
           dag_pipeline.DagOutput('output'): r}
    with self.assertRaises(dag_pipeline.InvalidDAGError):
      dag_pipeline.DAGPipeline(dag)

    dag = {q: dag_pipeline.DagInput(Type0),
           r: UnitQ,
           dag_pipeline.DagOutput('output'): r}
    with self.assertRaises(dag_pipeline.InvalidDAGError):
      dag_pipeline.DAGPipeline(dag)

    dag = {q: dag_pipeline.DagInput(Type0),
           r: 123,
           dag_pipeline.DagOutput('output'): r}
    with self.assertRaises(dag_pipeline.InvalidDAGError):
      dag_pipeline.DAGPipeline(dag)

    dag = {dag_pipeline.DagInput(Type0): q,
           dag_pipeline.DagOutput(): q}
    with self.assertRaises(dag_pipeline.InvalidDAGError):
      dag_pipeline.DAGPipeline(dag)

    dag = {q: dag_pipeline.DagInput(Type0),
           q: dag_pipeline.DagOutput('output')}
    with self.assertRaises(dag_pipeline.InvalidDAGError):
      dag_pipeline.DAGPipeline(dag)

    dag = {q: dag_pipeline.DagInput(Type0),
           r: {'abc': q['a'], 'def': 123},
           dag_pipeline.DagOutput('output'): r}
    with self.assertRaises(dag_pipeline.InvalidDAGError):
      dag_pipeline.DAGPipeline(dag)

    dag = {q: dag_pipeline.DagInput(Type0),
           r: {123: q['a']},
           dag_pipeline.DagOutput('output'): r}
    with self.assertRaises(dag_pipeline.InvalidDAGError):
      dag_pipeline.DAGPipeline(dag)

  def testTypeMismatchError(self):
    class UnitQ(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, Type1)

      def transform(self, input_object):
        pass

    class UnitR(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type1, {'a': Type2, 'b': Type3})

      def transform(self, input_object):
        pass

    class UnitS(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, {'x': Type2, 'y': Type3}, Type4)

      def transform(self, input_object):
        pass

    class UnitT(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, {'x': Type2, 'y': Type5}, Type4)

      def transform(self, input_object):
        pass

    q, r, s, t = UnitQ(), UnitR(), UnitS(), UnitT()
    dag = {q: dag_pipeline.DagInput(Type1),
           r: q,
           s: r,
           dag_pipeline.DagOutput('output'): s}
    with self.assertRaises(dag_pipeline.TypeMismatchError):
      dag_pipeline.DAGPipeline(dag)

    q2 = UnitQ()
    dag = {q: dag_pipeline.DagInput(Type0),
           q2: q,
           dag_pipeline.DagOutput('output'): q2}
    with self.assertRaises(dag_pipeline.TypeMismatchError):
      dag_pipeline.DAGPipeline(dag)

    dag = {q: dag_pipeline.DagInput(Type0),
           r: q,
           s: {'x': r['b'], 'y': r['a']},
           dag_pipeline.DagOutput('output'): s}
    with self.assertRaises(dag_pipeline.TypeMismatchError):
      dag_pipeline.DAGPipeline(dag)

    dag = {q: dag_pipeline.DagInput(Type0),
           r: q,
           t: r,
           dag_pipeline.DagOutput('output'): t}
    with self.assertRaises(dag_pipeline.TypeMismatchError):
      dag_pipeline.DAGPipeline(dag)

  def testDependencyLoops(self):
    class UnitQ(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, Type1)

      def transform(self, input_object):
        pass

    class UnitR(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type1, Type0)

      def transform(self, input_object):
        pass

    class UnitS(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, {'a': Type1, 'b': Type0}, Type1)

      def transform(self, input_object):
        pass

    class UnitT(pipeline.Pipeline):

      def __init__(self, name='UnitT'):
        pipeline.Pipeline.__init__(self, Type0, Type0, name)

      def transform(self, input_object):
        pass

    q, r, s, t = UnitQ(), UnitR(), UnitS(), UnitT()
    dag = {q: dag_pipeline.DagInput(q.input_type),
           s: {'a': q, 'b': r},
           r: s,
           dag_pipeline.DagOutput('output'): r,
           dag_pipeline.DagOutput('output_2'): s}
    with self.assertRaises(dag_pipeline.BadTopologyError):
      dag_pipeline.DAGPipeline(dag)

    dag = {s: {'a': dag_pipeline.DagInput(Type1), 'b': r},
           r: s,
           dag_pipeline.DagOutput('output'): r}
    with self.assertRaises(dag_pipeline.BadTopologyError):
      dag_pipeline.DAGPipeline(dag)

    dag = {dag_pipeline.DagOutput('output'): dag_pipeline.DagInput(Type0),
           t: t}
    with self.assertRaises(dag_pipeline.BadTopologyError):
      dag_pipeline.DAGPipeline(dag)

    t2 = UnitT('UnitT2')
    dag = {dag_pipeline.DagOutput('output'): dag_pipeline.DagInput(Type0),
           t2: t,
           t: t2}
    with self.assertRaises(dag_pipeline.BadTopologyError):
      dag_pipeline.DAGPipeline(dag)

  def testDisjointGraph(self):
    class UnitQ(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, Type1)

      def transform(self, input_object):
        pass

    class UnitR(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type1, {'a': Type2, 'b': Type3})

      def transform(self, input_object):
        pass

    q, r = UnitQ(), UnitR()
    dag = {q: dag_pipeline.DagInput(q.input_type),
           dag_pipeline.DagOutput(): r}
    with self.assertRaises(dag_pipeline.NotConnectedError):
      dag_pipeline.DAGPipeline(dag)

    q, r = UnitQ(), UnitR()
    dag = {q: dag_pipeline.DagInput(q.input_type),
           dag_pipeline.DagOutput(): {'a': q, 'b': r['b']}}
    with self.assertRaises(dag_pipeline.NotConnectedError):
      dag_pipeline.DAGPipeline(dag)

    # Pipelines that do not output to anywhere are not allowed.
    dag = {dag_pipeline.DagOutput('output'):
               dag_pipeline.DagInput(q.input_type),
           q: dag_pipeline.DagInput(q.input_type),
           r: q}
    with self.assertRaises(dag_pipeline.NotConnectedError):
      dag_pipeline.DAGPipeline(dag)

    # Pipelines which need to be executed but don't have inputs are not allowed.
    dag = {dag_pipeline.DagOutput('output'):
               dag_pipeline.DagInput(q.input_type),
           r: q,
           dag_pipeline.DagOutput(): r}
    with self.assertRaises(dag_pipeline.NotConnectedError):
      dag_pipeline.DAGPipeline(dag)

  def testBadInputOrOutputError(self):
    class UnitQ(pipeline.Pipeline):

      def __init__(self, name='UnitQ'):
        pipeline.Pipeline.__init__(self, Type0, Type1, name)

      def transform(self, input_object):
        pass

    class UnitR(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type1, Type0)

      def transform(self, input_object):
        pass

    # Missing Input.
    q, r = UnitQ(), UnitR()
    dag = {r: q,
           dag_pipeline.DagOutput('output'): r}
    with self.assertRaises(dag_pipeline.BadInputOrOutputError):
      dag_pipeline.DAGPipeline(dag)

    # Missing Output.
    dag = {q: dag_pipeline.DagInput(Type0),
           r: q}
    with self.assertRaises(dag_pipeline.BadInputOrOutputError):
      dag_pipeline.DAGPipeline(dag)

    # Multiple instances of Input with the same type IS allowed.
    q2 = UnitQ('UnitQ2')
    dag = {q: dag_pipeline.DagInput(Type0),
           q2: dag_pipeline.DagInput(Type0),
           dag_pipeline.DagOutput(): {'q': q, 'q2': q2}}
    _ = dag_pipeline.DAGPipeline(dag)

    # Multiple instances with different types is not allowed.
    dag = {q: dag_pipeline.DagInput(Type0),
           r: dag_pipeline.DagInput(Type1),
           dag_pipeline.DagOutput(): {'q': q, 'r': r}}
    with self.assertRaises(dag_pipeline.BadInputOrOutputError):
      dag_pipeline.DAGPipeline(dag)

  def testDuplicateNameError(self):

    class UnitQ(pipeline.Pipeline):

      def __init__(self, name='UnitQ'):
        pipeline.Pipeline.__init__(self, Type0, Type1, name)

      def transform(self, input_object):
        pass

    q, q2 = UnitQ(), UnitQ()
    dag = {q: dag_pipeline.DagInput(Type0),
           q2: dag_pipeline.DagInput(Type0),
           dag_pipeline.DagOutput(): {'q': q, 'q2': q2}}
    with self.assertRaises(dag_pipeline.DuplicateNameError):
      dag_pipeline.DAGPipeline(dag)

  def testInvalidDictionaryOutputError(self):
    b = UnitB()
    dag = {b: dag_pipeline.DagInput(b.input_type),
           dag_pipeline.DagOutput(): b}
    with self.assertRaises(dag_pipeline.InvalidDictionaryOutputError):
      dag_pipeline.DAGPipeline(dag)

    a = UnitA()
    dag = {a: dag_pipeline.DagInput(b.input_type),
           dag_pipeline.DagOutput('output'): a}
    with self.assertRaises(dag_pipeline.InvalidDictionaryOutputError):
      dag_pipeline.DAGPipeline(dag)

    a2 = UnitA()
    dag = {a: dag_pipeline.DagInput(a.input_type),
           a2: dag_pipeline.DagInput(a2.input_type),
           dag_pipeline.DagOutput('output'): {'t1': a['t1'], 't2': a2['t2']}}
    with self.assertRaises(dag_pipeline.InvalidDictionaryOutputError):
      dag_pipeline.DAGPipeline(dag)

  def testInvalidTransformOutputError(self):
    # This happens when the output of a pipeline's `transform` method does not
    # match the type signature given by the pipeline's `output_type`.

    class UnitQ1(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, Type1)

      def transform(self, input_object):
        return [Type2(1)]

    class UnitQ2(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, Type1)

      def transform(self, input_object):
        return [Type1(1, 2), Type2(1)]

    class UnitQ3(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, Type1)

      def transform(self, input_object):
        return Type1(1, 2)

    class UnitR1(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, {'xy': Type1, 'z': Type2})

      def transform(self, input_object):
        return {'xy': [Type1(1, 2)], 'z': [Type1(1, 2)]}

    class UnitR2(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, {'xy': Type1, 'z': Type2})

      def transform(self, input_object):
        return {'xy': [Type1(1, 2)]}

    class UnitR3(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, {'xy': Type1, 'z': Type2})

      def transform(self, input_object):
        return [{'xy': [Type1(1, 2)], 'z': Type2(1)}]

    class UnitR4(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, {'xy': Type1, 'z': Type2})

      def transform(self, input_object):
        return [{'xy': [Type1(1, 2), Type2(1)], 'z': [Type2(1)]}]

    class UnitR5(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, Type0, {'xy': Type1, 'z': Type2})

      def transform(self, input_object):
        return [{'xy': [Type1(1, 2), Type1(1, 3)], 'z': [Type2(1)], 'q': []}]

    for pipeline_class in [UnitQ1, UnitQ2, UnitQ3,
                           UnitR1, UnitR2, UnitR3, UnitR4, UnitR5]:
      pipe = pipeline_class()
      if pipeline_class.__name__.startswith('UnitR'):
        output = dag_pipeline.DagOutput()
      else:
        output = dag_pipeline.DagOutput('output')
      dag = {pipe: dag_pipeline.DagInput(pipe.input_type),
             output: pipe}
      dag_pipe_obj = dag_pipeline.DAGPipeline(dag)
      with self.assertRaises(dag_pipeline.InvalidTransformOutputError):
        dag_pipe_obj.transform(Type0(1, 2, 3))

  def testInvalidStatisticsError(self):
    class UnitQ(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, str, str)

      def transform(self, input_object):
        self._set_stats([statistics.Counter('stat_1', 5), 1234])
        return [input_object]

    class UnitR(pipeline.Pipeline):

      def __init__(self):
        pipeline.Pipeline.__init__(self, str, str)

      def transform(self, input_object):
        self._set_stats(statistics.Counter('stat_1', 5))
        return [input_object]

    q = UnitQ()
    dag = {q: dag_pipeline.DagInput(q.input_type),
           dag_pipeline.DagOutput('output'): q}
    dag_pipe_obj = dag_pipeline.DAGPipeline(dag)
    with self.assertRaises(pipeline.InvalidStatisticsError):
      dag_pipe_obj.transform('hello world')

    r = UnitR()
    dag = {r: dag_pipeline.DagInput(q.input_type),
           dag_pipeline.DagOutput('output'): r}
    dag_pipe_obj = dag_pipeline.DAGPipeline(dag)
    with self.assertRaises(pipeline.InvalidStatisticsError):
      dag_pipe_obj.transform('hello world')


if __name__ == '__main__':
  absltest.main()
