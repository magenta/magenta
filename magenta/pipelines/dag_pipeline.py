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
"""Pipeline that runs arbitrary pipelines composed in a graph."""


import itertools

# internal imports
from magenta.pipelines import pipeline
from magenta.pipelines import statistics


class Output(object):

  def __init__(self, name=None):
    self.name = name
    self.output_type = None
    self.input_type = None

  def __eq__(self, other):
    return isinstance(other, Output) and other.name == self.name

  def __hash__(self):
    return hash(self.name)

  def __repr__(self):
    return 'Output(%s)' % self.name
    

class Input(object):

  def __init__(self, type_):
    self.output_type = type_

  def __eq__(self, other):
    return isinstance(other, Input) and other.output_type == self.output_type

  def __hash__(self):
    return hash(self.output_type)

  def __repr__(self):
    return 'Input(%s)' % self.output_type


def join_lists_or_dicts(inputs):
  if not inputs:
    return []
  if isinstance(inputs[0], dict):
    concated = dict([(key, list()) for key in inputs[0].keys()])
    for d in inputs:
      assert isinstance(d, dict)
      assert d.keys() == concated.keys()
      for k, val in d.items():
        assert isinstance(val, list)
        concated[k] += val
  else:
    concated = []
    for l in inputs:
      assert isinstance(l, list)
      concated += l
  return concated


def cartesian_product(*input_lists):
  return itertools.product(*input_lists)


class DAGPipeline(pipeline.Pipeline):
  
  def __init__(self, dag):
    # Expand DAG shorthand. Currently the only shorthand is direct connections.
    # A direct connection is a connection {a: b} where a.input_type is a dict
    # and b.output_type is a dict, and a.input_type == b.output_type.
    # {a: b} is expanded to
    # {a: {"name_1": b["name_1"], "name_2": b["name_2"], ...}}.
    # {Output(): {"name_1", obj1, "name_2": obj2, ...} is expanded to
    # {Output("name_1"): obj1, Output("name_2"): obj2, ...}.
    self.dag = dict(self._expand_dag_shorthands(dag))

    self.outputs = [unit for unit in self.dag if isinstance(unit, Output)]
    self.output_names = dict([(output.name, output) for output in self.outputs])
    inputs = set()
    for deps in self.dag.values():
      units = self._get_units(deps)
      for unit in units:
        if isinstance(unit, Input):
          inputs.add(unit)
    assert len(inputs) == 1
    self.input = inputs.pop()
    self.stats = {}
    
    output_signature = dict([(output.name, self.dag[output].output_type) for output in self.outputs])
    super(DAGPipeline, self).__init__(input_type=self.input.output_type, output_type=output_signature)
    
    # TODO: allow Output(): {'output_1': obj_1, 'output_2': obj_2, ...}
    # TODO: allow direct connection, i.e. if a.output_type is a dict, and b.input_type is a dict,
    #    then {a: b} connects their inputs and outputs.
    
    # TODO: Make sure DAG is valid.
    # Input types match output types. No cycles (cycles will be found below). Nothing depends on outputs. Things that require input get input.
    for unit, dependency in self.dag.items():
      if not self._is_valid_dependency(dependency):
        raise ValueError('Invalid dependency %s: %s' % (unit, dependency))
    
    # Construct topological ordering.
    # https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm
    graph = dict([(unit, [self._get_units(self.dag[unit]), 0]) for unit in self.dag])
    graph[self.input] = [[], 0]
    for unit, (forward_connections, _) in graph.items():
      for to_unit in forward_connections:
        graph[to_unit][1] += 1
    self.call_list = call_list = [] # Contains topologically sorted elements.
    nodes = set(self.outputs)
    while nodes:
      n = nodes.pop()
      call_list.append(n)
      for m in graph[n][0]:
        graph[m][1] -= 1
        if graph[m][1] == 0:
          nodes.add(m)
        elif graph[m][1] < 0:
          raise Exception('Bug')
    # Check if any edges remain.
    for unit in call_list:
      if graph[unit][1] != 0:
        raise Exception('Dependency loop found on %s' % unit)
    call_list.reverse()
    assert call_list[0] == self.input

  def _expand_dag_shorthands(self, dag):
    for key, val in dag.items():
      # Direct connection.
      if (isinstance(key, pipeline.Pipeline) and
          isinstance(val, pipeline.Pipeline) and
          isinstance(key.input_type, dict) and
          key.input_type == val.output_type):
        yield key, dict([(name, val[name]) for name in val.output_type])
      elif key == Output():
        if (isinstance(val, pipeline.Pipeline) and
            isinstance(val.output_type, dict)):
          dependencies = [(name, val[name]) for name in val.output_type]
        elif isinstance(val, dict):
          dependencies = val.items()
        else:
          raise ValueError('Output() with no name can only be connected to a dictionary or a Pipeline whose output_type is a dictionary. Found Output() connected to %s' % val)
        for name, dep in dependencies:
          yield Output(name), dep
      else:
        yield key, val

          
  def _get_units(self, dependencies):
    dep_list = []
    if isinstance(dependencies, dict):
      dep_list.extend(dependencies.values())
    else:
      dep_list.append(dependencies)
    return [self._get_unit(punit) for punit in dep_list]
  
  def _get_unit(self, possible_unit):
    if isinstance(possible_unit, pipeline.Pipeline):
      return possible_unit
    if isinstance(possible_unit, pipeline.Key):
      assert isinstance(possible_unit.unit, pipeline.Pipeline)
      return possible_unit.unit
    if isinstance(possible_unit, Input):
      return possible_unit
    raise Exception()
  
  def _is_valid_dependency(self, dependency):
    if isinstance(dependency, dict):
      values = dependency.values()
    else:
      values = [dependency]
    for v in values:
      if not (isinstance(v, pipeline.Pipeline) or
              (isinstance(v, pipeline.Key) and
               isinstance(v.unit, pipeline.Pipeline)) or
              isinstance(v, Input)):
        return False
    return True
  
  def transform(self, input_object):
    """Runs the pipeline on the given input.

    Subclasses must implement this method.

    Args:
      input_object: Any object. The required type depends on implementation.

    Returns:
      A dictionary mapping output names to lists of objects. The object types
      depend on implementation. Each output name corresponds to an output
      collection. See get_output_names method.
    """
    self.stats = {}
    results = {self.input: [input_object]}
    for unit in self.call_list[1:]:
      # TODO: assert that output types are expected. assert that stat objects are valid.
      # compute transformation.

      if isinstance(unit, Output):
        unit_outputs = self._get_outputs_as_signature(self.dag[unit], results)
      else:
        unit_inputs = self._get_inputs_for_unit(unit, results)
        if not unit_inputs:
          # If this unit has no inputs don't run it.
          results[unit] = []
          continue

        merged_stats = {}
        def stats_accumulator():
          for single_input in unit_inputs:
            results_ = unit.transform(single_input)
            statistics.merge_statistics_dicts(merged_stats, unit.get_stats())
            yield results_

        unjoined_outputs = list(stats_accumulator())
        unit_outputs = join_lists_or_dicts(unjoined_outputs)
      results[unit] = unit_outputs
      
      # Merge statistics.
      if isinstance(unit, pipeline.Pipeline):
        for stat_name, stat_value in merged_stats.items():
          full_name = '%s_%s' % (type(unit).__name__, stat_name)
          assert full_name not in self.stats
          self.stats[full_name] = stat_value
    return dict([(output.name, results[output]) for output in self.outputs])
  
  def _get_outputs_as_signature(self, signature, outputs):
    def _get_outputs_for_key(unit_or_key, outputs):
      if isinstance(unit_or_key, pipeline.Key):
        if not outputs[unit_or_key.unit]:
          # If there are no outputs, just return nothing.
          return outputs[unit_or_key.unit]
        assert isinstance(outputs[unit_or_key.unit], dict)
        return outputs[unit_or_key.unit][unit_or_key.key]
      assert isinstance(unit_or_key, (pipeline.Pipeline, Input))
      return outputs[unit_or_key]
    if isinstance(signature, dict):
      return dict([(name, _get_outputs_for_key(unit_or_key, outputs)) for name, unit_or_key in signature.items()])
    return _get_outputs_for_key(signature, outputs)
  
  def _get_inputs_for_unit(self, unit, results, list_operation=cartesian_product):
    signature = self.dag[unit]
    previous_outputs = self._get_outputs_as_signature(self.dag[unit], results)
    
    if isinstance(previous_outputs, dict):
      names = list(previous_outputs.keys())
      lists = [previous_outputs[name] for name in names]
      stack = list_operation(*lists)
      return [dict(zip(names, values)) for values in stack]
    else:
      return previous_outputs

  def get_stats(self):
    """Returns statistics about pipeline runs.

    Call after running transform to get statistics about it.

    Returns:
      Dictionary mapping statistic name to statistic value.
    """
    return self.stats
    