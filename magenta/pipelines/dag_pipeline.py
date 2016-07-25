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


class Output(object):

  def __init__(self, name):
    self.name = name
    self.output_type = None
    self.input_type = None
    

class Input(object):

  def __init__(self, type_):
    self.output_type = type_

  def __eq__(self, other):
    return isinstance(other, Input) and other.output_type == self.output_type

  def __hash__(self):
    return hash(self.output_type)
  

def map_and_flatten(input_list, func):
  return [output
          for single_input in input_list
          for output in func(single_input)]


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
    self.dag = dag
    self.outputs = [unit for unit in dag if isinstance(unit, Output)]
    self.output_names = dict([(output.name, output) for output in self.outputs])
    inputs = set()
    for deps in dag.values():
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
    # Use https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
    
    # Construct topological ordering.
    # https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm
    graph = dict([(unit, [self._get_units(dag[unit]), 0]) for unit in dag])
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
          
  def _get_units(self, dependencies):
    dep_list = []
    if isinstance(dependencies, dict):
      dep_list.extend(dependencies.values())
    elif isinstance(dependencies, (tuple, list)):
      dep_list.extend(dependencies)
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
    elif isinstance(dependency, (tuple, list)):
      values = dependency
    else:
      values = [dependency]
    for v in values:
      if not (isinstance(v, PipelineUnit) or (isinstance(v, pipeline.Key) and isinstance(v.unit, PipelineUnit)) or isinstance(v, Input)):
        return False
    return True
  
  def _type_signature(self, dependency):
    if isinstance(dependency, dict):
      return dict([(key, value.output_type) for (key, value) in dependency.items()])
    elif isinstance(dependency, (tuple, list)):
      return tuple([value.output_type for value in dependency])
    else:
      return dependency.output_type
  
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
        unit_outputs = join_lists_or_dicts([unit.transform(single_input) for single_input in unit_inputs])
      results[unit] = unit_outputs
      
      # get stats.
      if isinstance(unit, pipeline.Pipeline):
        for stat_name, stat_value in unit.get_stats().items():
          full_name = '%s_%s' % (type(unit).__name__, stat_name)
          assert full_name not in self.stats
          self.stats[full_name] = stat_value
    return dict([(output.name, results[output]) for output in self.outputs])
  
  def _get_outputs_as_signature(self, signature, outputs):
    def _get_outputs_for_key(unit_or_key, outputs):
      if isinstance(unit_or_key, pipeline.Key):
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
    