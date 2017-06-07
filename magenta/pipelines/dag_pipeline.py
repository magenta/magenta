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
"""Pipeline that runs arbitrary pipelines composed in a graph.

Some terminology used in the code.

dag: Directed acyclic graph.
unit: A Pipeline which is run inside DAGPipeline.
connection: A key value pair in the DAG dictionary.
dependency: The right hand side (value in key value dictionary pair) of a DAG
    connection. Can be a Pipeline, DagInput, PipelineKey, or dictionary mapping
    names to one of those.
subordinate: Any DagInput, Pipeline, or PipelineKey object that appears in a
    dependency.
shorthand: Invalid things that can be put in the DAG which get converted to
    valid things before parsing. These things are for convenience.
type signature: Something that can be returned from Pipeline's `output_type`
    or `input_type`. A python class, or dictionary mapping names to classes.
"""


import itertools

# internal imports
from magenta.pipelines import pipeline


class DagOutput(object):
  """Represents an output destination for a `DAGPipeline`.

  Each `DagOutput(name)` instance given to DAGPipeline will
  be a final output bucket with the same name. If writing
  output buckets to disk, the names become dataset names.

  The name can be omitted if connecting `DagOutput()` to a
  dictionary mapping names to pipelines.
  """

  def __init__(self, name=None):
    """Create an `DagOutput` with the given name.

    Args:
      name: If given, a string name which defines the name of this output.
          If not given, the names in the dictionary this is connected to
          will be used as output names.
    """
    self.name = name

    # `output_type` and `input_type` are set by DAGPipeline. Since `DagOutput`
    # is not given its type, the type must be infered from what it is connected
    # to in the DAG. Having `output_type` and `input_type` makes `DagOutput` act
    # like a `Pipeline` in some cases.
    self.output_type = None
    self.input_type = None

  def __eq__(self, other):
    return isinstance(other, DagOutput) and other.name == self.name

  def __hash__(self):
    return hash(self.name)

  def __repr__(self):
    return 'DagOutput(%s)' % self.name


class DagInput(object):
  """Represents an input source for a `DAGPipeline`.

  Give an `DagInput` instance to `DAGPipeline` by connecting `Pipeline` objects
  to it in the DAG.

  When `DAGPipeline.transform` is called, the input object
  will be fed to any Pipeline instances connected to an
  `DagInput` given in the DAG.

  The type given to `DagInput` will be the `DAGPipeline`'s `input_type`.
  """

  def __init__(self, type_):
    """Create an `DagInput` with the given type.

    Args:
      type_: The Python class which inputs to `DAGPipeline` should be
          instances of. `DAGPipeline.input_type` will be this type.
    """
    self.output_type = type_

  def __eq__(self, other):
    return isinstance(other, DagInput) and other.output_type == self.output_type

  def __hash__(self):
    return hash(self.output_type)

  def __repr__(self):
    return 'DagInput(%s)' % self.output_type


def _all_are_type(elements, target_type):
  """Checks that all the given elements are the target type.

  Args:
    elements: A list of objects.
    target_type: The Python class which all elemenets need to be an instance of.

  Returns:
    True if every object in `elements` is an instance of `target_type`, and
    False otherwise.
  """
  return all(isinstance(elem, target_type) for elem in elements)


class InvalidDAGException(Exception):
  """Thrown when the DAG dictionary is not well formatted.

  This can be because a `destination: dependency` pair is not in the form
  `Pipeline: Pipeline` or `Pipeline: {'name_1': Pipeline, ...}` (Note that
  Pipeline or PipelineKey objects both are allowed in the dependency). It is
  also thrown when `DagInput` is given as a destination, or `DagOutput` is given
  as a dependency.
  """
  pass


class DuplicateNameException(Exception):
  """Thrown when two `Pipeline` instances in the DAG have the same name.

  Pipeline names will be used as name spaces for the statistics they produce
  and we don't want any conflicts.
  """
  pass


class BadTopologyException(Exception):
  """Thrown when there is a directed cycle."""
  pass


class NotConnectedException(Exception):
  """Thrown when the DAG is disconnected somewhere.

  Either because a `Pipeline` used in a dependency has nothing feeding into it,
  or because a `Pipeline` given as a destination does not feed anywhere.
  """
  pass


class TypeMismatchException(Exception):
  """Thrown when type signatures in a connection don't match.

  In the DAG's `destination: dependency` pairs, type signatures must match.
  """
  pass


class BadInputOrOutputException(Exception):
  """Thrown when `DagInput` or `DagOutput` are not used in the graph correctly.

  Specifically when there are no `DagInput` objects, more than one `DagInput`
  with different types, or there is no `DagOutput` object.
  """
  pass


class InvalidDictionaryOutput(Exception):
  """Thrown when `DagOutput` and dictionaries are not used correctly.

  Specifically when `DagOutput()` is used without a dictionary dependency, or
  `DagOutput(name)` is used with a `name` and with a dictionary dependency.
  """
  pass


class InvalidTransformOutputException(Exception):
  """Thrown when a Pipeline does not output types matching its `output_type`.
  """
  pass


class DAGPipeline(pipeline.Pipeline):
  """A directed acyclic graph pipeline.

  This Pipeline can be given an arbitrary graph composed of Pipeline instances
  and will run all of those pipelines feeding outputs to inputs. See README.md
  for details.

  Use DAGPipeline to compose multiple smaller pipelines together.
  """

  def __init__(self, dag, pipeline_name='DAGPipeline'):
    """Constructs a DAGPipeline.

    A DAG (direct acyclic graph) is given which fully specifies what the
    DAGPipeline runs.

    Args:
      dag: A dictionary mapping `Pipeline` or `DagOutput` instances to any of
         `Pipeline`, `PipelineKey`, `DagInput`. `dag` defines a directed acyclic
         graph.
      pipeline_name: String name of this Pipeline object.

    Raises:
      InvalidDAGException: If each key value pair in the `dag` dictionary is
          not of the form
          (Pipeline or DagOutput): (Pipeline, PipelineKey, or DagInput).
      TypeMismatchException: The type signature of each key and value in `dag`
          must match, otherwise this will be thrown.
      DuplicateNameException: If two `Pipeline` instances in `dag` have the
          same string name.
      BadInputOrOutputException: If there are no `DagOutput` instaces in `dag`
          or not exactly one `DagInput` plus type combination in `dag`.
      InvalidDictionaryOutput: If `DagOutput()` is not connected to a
          dictionary, or `DagOutput(name)` is not connected to a Pipeline,
          PipelineKey, or DagInput instance.
      NotConnectedException: If a `Pipeline` used in a dependency has nothing
          feeding into it, or a `Pipeline` used as a destination does not feed
          anywhere.
      BadTopologyException: If there there is a directed cycle in `dag`.
      Exception: Misc. exceptions.
    """
    # Expand DAG shorthand.
    self.dag = dict(self._expand_dag_shorthands(dag))

    # Make sure DAG is valid.
    # DagInput types match output types. Nothing depends on outputs.
    # Things that require input get input. DAG is composed of correct types.
    for unit, dependency in self.dag.items():
      if not isinstance(unit, (pipeline.Pipeline, DagOutput)):
        raise InvalidDAGException(
            'Dependency {%s: %s} is invalid. Left hand side value %s must '
            'either be a Pipeline or DagOutput object'
            % (unit, dependency, unit))
      if isinstance(dependency, dict):
        if not all([isinstance(name, basestring) for name in dependency]):
          raise InvalidDAGException(
              'Dependency {%s: %s} is invalid. Right hand side keys %s must be '
              'strings' % (unit, dependency, dependency.keys()))
        values = dependency.values()
      else:
        values = [dependency]
      for subordinate in values:
        if not (isinstance(subordinate, pipeline.Pipeline) or
                (isinstance(subordinate, pipeline.PipelineKey) and
                 isinstance(subordinate.unit, pipeline.Pipeline)) or
                isinstance(subordinate, DagInput)):
          raise InvalidDAGException(
              'Dependency {%s: %s} is invalid. Right hand side subordinate %s '
              'must be either a Pipeline, PipelineKey, or DagInput object'
              % (unit, dependency, subordinate))

      # Check that all input types match output types.
      if isinstance(unit, DagOutput):
        # DagOutput objects don't know their types.
        continue
      if unit.input_type != self._get_type_signature_for_dependency(dependency):
        raise TypeMismatchException(
            'Invalid dependency {%s: %s}. Required `input_type` of left hand '
            'side is %s. DagOutput type of right hand side is %s.'
            % (unit, dependency, unit.input_type,
               self._get_type_signature_for_dependency(dependency)))

    # Make sure all Pipeline names are unique, so that Statistic objects don't
    # clash.
    sorted_unit_names = sorted(
        [(unit, unit.name) for unit in self.dag],
        key=lambda t: t[1])
    for index, (unit, name) in enumerate(sorted_unit_names[:-1]):
      if name == sorted_unit_names[index + 1][1]:
        other_unit = sorted_unit_names[index + 1][0]
        raise DuplicateNameException(
            'Pipelines %s and %s both have name "%s". Each Pipeline must have '
            'a unique name.' % (unit, other_unit, name))

    # Find DagInput and DagOutput objects and make sure they are being used
    # correctly.
    self.outputs = [unit for unit in self.dag if isinstance(unit, DagOutput)]
    self.output_names = dict([(output.name, output) for output in self.outputs])
    for output in self.outputs:
      output.input_type = output.output_type = (
          self._get_type_signature_for_dependency(self.dag[output]))
    inputs = set()
    for deps in self.dag.values():
      units = self._get_units(deps)
      for unit in units:
        if isinstance(unit, DagInput):
          inputs.add(unit)
    if len(inputs) != 1:
      if not inputs:
        raise BadInputOrOutputException(
            'No DagInput object found. DagInput is the start of the pipeline.')
      else:
        raise BadInputOrOutputException(
            'Multiple DagInput objects found. Only one input is supported.')
    if not self.outputs:
      raise BadInputOrOutputException(
          'No DagOutput objects found. DagOutput is the end of the pipeline.')
    self.input = inputs.pop()

    # Compute output_type for self and call super constructor.
    output_signature = dict([(output.name, output.output_type)
                             for output in self.outputs])
    super(DAGPipeline, self).__init__(
        input_type=self.input.output_type,
        output_type=output_signature,
        name=pipeline_name)

    # Make sure all Pipeline objects have DAG vertices that feed into them,
    # and feed their output into other DAG vertices.
    all_subordinates = (
        set([dep_unit for unit in self.dag
             for dep_unit in self._get_units(self.dag[unit])])
        .difference(set([self.input])))
    all_destinations = set(self.dag.keys()).difference(set(self.outputs))
    if all_subordinates != all_destinations:
      units_with_no_input = all_subordinates.difference(all_destinations)
      units_with_no_output = all_destinations.difference(all_subordinates)
      if units_with_no_input:
        raise NotConnectedException(
            '%s is given as a dependency in the DAG but has nothing connected '
            'to it. Nothing in the DAG feeds into it.'
            % units_with_no_input.pop())
      else:
        raise NotConnectedException(
            '%s is given as a destination in the DAG but does not output '
            'anywhere. It is a deadend.' % units_with_no_output.pop())

    # Construct topological ordering to determine the execution order of the
    # pipelines.
    # https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm

    # `graph` maps a pipeline to the pipelines it depends on. Each dict value
    # is a list with the dependency pipelines in the 0th position, and a count
    # of forward connections to the key pipeline (how many pipelines use this
    # pipeline as a dependency).
    graph = dict([(unit, [self._get_units(self.dag[unit]), 0])
                  for unit in self.dag])
    graph[self.input] = [[], 0]
    for unit, (forward_connections, _) in graph.items():
      for to_unit in forward_connections:
        graph[to_unit][1] += 1
    self.call_list = call_list = []  # Topologically sorted elements go here.
    nodes = set(self.outputs)
    while nodes:
      n = nodes.pop()
      call_list.append(n)
      for m in graph[n][0]:
        graph[m][1] -= 1
        if graph[m][1] == 0:
          nodes.add(m)
        elif graph[m][1] < 0:
          raise Exception(
              'Congratulations, you found a bug! Please report this issue at '
              'https://github.com/tensorflow/magenta/issues and copy/paste the '
              'following: dag=%s, graph=%s, call_list=%s' % (self.dag, graph,
                                                             call_list))
    # Check for cycles by checking if any edges remain.
    for unit in graph:
      if graph[unit][1] != 0:
        raise BadTopologyException('Dependency loop found on %s' % unit)

    # Note: this exception should never be raised. Disconnected graphs will be
    # caught where NotConnectedException is raised. If this exception goes off
    # there is likely a bug.
    if set(call_list) != set(
        list(all_subordinates) + self.outputs + [self.input]):
      raise BadTopologyException('Not all pipelines feed into an output or '
                                 'there is a dependency loop.')

    call_list.reverse()
    assert call_list[0] == self.input

  def _expand_dag_shorthands(self, dag):
    """Expand DAG shorthand.

    Currently the only shorthand is "direct connection".
    A direct connection is a connection {a: b} where a.input_type is a dict,
    b.output_type is a dict, and a.input_type == b.output_type. This is not
    actually valid, but we can convert it to a valid connection.

    {a: b} is expanded to
    {a: {"name_1": b["name_1"], "name_2": b["name_2"], ...}}.
    {DagOutput(): {"name_1", obj1, "name_2": obj2, ...} is expanded to
    {DagOutput("name_1"): obj1, DagOutput("name_2"): obj2, ...}.

    Args:
      dag: A dictionary encoding the DAG.

    Yields:
      Key, value pairs for a new dag dictionary.

    Raises:
      InvalidDictionaryOutput: If `DagOutput` is not used correctly.
    """
    for key, val in dag.items():
      # Direct connection.
      if (isinstance(key, pipeline.Pipeline) and
          isinstance(val, pipeline.Pipeline) and
          isinstance(key.input_type, dict) and
          key.input_type == val.output_type):
        yield key, dict([(name, val[name]) for name in val.output_type])
      elif key == DagOutput():
        if (isinstance(val, pipeline.Pipeline) and
            isinstance(val.output_type, dict)):
          dependency = [(name, val[name]) for name in val.output_type]
        elif isinstance(val, dict):
          dependency = val.items()
        else:
          raise InvalidDictionaryOutput(
              'DagOutput() with no name can only be connected to a dictionary '
              'or a Pipeline whose output_type is a dictionary. Found '
              'DagOutput() connected to %s' % val)
        for name, subordinate in dependency:
          yield DagOutput(name), subordinate
      elif isinstance(key, DagOutput):
        if isinstance(val, dict):
          raise InvalidDictionaryOutput(
              'DagOutput("%s") which has name "%s" can only be connected to a '
              'single input, not dictionary %s. Use DagOutput() without name '
              'instead.' % (key.name, key.name, val))
        if (isinstance(val, pipeline.Pipeline) and
            isinstance(val.output_type, dict)):
          raise InvalidDictionaryOutput(
              'DagOutput("%s") which has name "%s" can only be connected to a '
              'single input, not pipeline %s which has dictionary '
              'output_type %s. Use DagOutput() without name instead.'
              % (key.name, key.name, val, val.output_type))
        yield key, val
      else:
        yield key, val

  def _get_units(self, dependency):
    """Gets list of units from a dependency."""
    dep_list = []
    if isinstance(dependency, dict):
      dep_list.extend(dependency.values())
    else:
      dep_list.append(dependency)
    return [self._validate_subordinate(sub) for sub in dep_list]

  def _validate_subordinate(self, subordinate):
    """Verifies that subordinate is DagInput, PipelineKey, or Pipeline."""
    if isinstance(subordinate, pipeline.Pipeline):
      return subordinate
    if isinstance(subordinate, pipeline.PipelineKey):
      if not isinstance(subordinate.unit, pipeline.Pipeline):
        raise InvalidDAGException(
            'PipelineKey object %s does not have a valid Pipeline'
            % subordinate)
      return subordinate.unit
    if isinstance(subordinate, DagInput):
      return subordinate
    raise InvalidDAGException(
        'Looking for Pipeline, PipelineKey, or DagInput object, but got %s'
        % type(subordinate))

  def _get_type_signature_for_dependency(self, dependency):
    """Gets the type signature of the dependency output."""
    if isinstance(dependency,
                  (pipeline.Pipeline, pipeline.PipelineKey, DagInput)):
      return dependency.output_type
    return dict([(name, sub_dep.output_type)
                 for name, sub_dep in dependency.items()])

  def transform(self, input_object):
    """Runs the DAG on the given input.

    All pipelines in the DAG will run.

    Args:
      input_object: Any object. The required type depends on implementation.

    Returns:
      A dictionary mapping output names to lists of objects. The object types
      depend on implementation. Each output name corresponds to an output
      collection. See get_output_names method.
    """
    def stats_accumulator(unit, unit_inputs, cumulative_stats):
      for single_input in unit_inputs:
        results_ = unit.transform(single_input)
        stats = unit.get_stats()
        cumulative_stats.extend(stats)
        yield results_

    stats = []
    results = {self.input: [input_object]}
    for unit in self.call_list[1:]:
      # Compute transformation.

      if isinstance(unit, DagOutput):
        unit_outputs = self._get_outputs_as_signature(self.dag[unit], results)
      else:
        unit_inputs = self._get_inputs_for_unit(unit, results)
        if not unit_inputs:
          # If this unit has no inputs don't run it.
          results[unit] = []
          continue

        unjoined_outputs = list(
            stats_accumulator(unit, unit_inputs, stats))
        unit_outputs = self._join_lists_or_dicts(unjoined_outputs, unit)
      results[unit] = unit_outputs

    self._set_stats(stats)
    return dict([(output.name, results[output]) for output in self.outputs])

  def _get_outputs_as_signature(self, dependency, outputs):
    """Returns a list or dict which matches the type signature of dependency.

    Args:
      dependency: DagInput, PipelineKey, Pipeline instance, or dictionary
          mapping names to those values.
      outputs: A database of computed unit outputs. A dictionary mapping
          Pipeline to list of objects.

    Returns:
      A list or dictionary of computed unit outputs which matches the type
      signature of the given dependency.
    """
    def _get_outputs_for_key(unit_or_key, outputs):
      if isinstance(unit_or_key, pipeline.PipelineKey):
        if not outputs[unit_or_key.unit]:
          # If there are no outputs, just return nothing.
          return outputs[unit_or_key.unit]
        assert isinstance(outputs[unit_or_key.unit], dict)
        return outputs[unit_or_key.unit][unit_or_key.key]
      assert isinstance(unit_or_key, (pipeline.Pipeline, DagInput))
      return outputs[unit_or_key]
    if isinstance(dependency, dict):
      return dict([(name, _get_outputs_for_key(unit_or_key, outputs))
                   for name, unit_or_key in dependency.items()])
    return _get_outputs_for_key(dependency, outputs)

  def _get_inputs_for_unit(self, unit, results,
                           list_operation=itertools.product):
    """Creates valid inputs for the given unit from the outputs in `results`.

    Args:
      unit: The `Pipeline` to create inputs for.
      results: A database of computed unit outputs. A dictionary mapping
          Pipeline to list of objects.
      list_operation: A function that maps lists of inputs to a single list of
          tuples, where each tuple is an input. This is used when `unit` takes
          a dictionary as input. Each tuple is used as the values for a
          dictionary input. This can be thought of as taking a sort of
          transpose of a ragged 2D array.
          The default is `itertools.product` which takes the cartesian product
          of the input lists.

    Returns:
      If `unit` takes a single input, a list of objects.
      If `unit` takes a dictionary input, a list of dictionaries each mapping
      string name to object.
    """
    previous_outputs = self._get_outputs_as_signature(self.dag[unit], results)

    if isinstance(previous_outputs, dict):
      names = list(previous_outputs.keys())
      lists = [previous_outputs[name] for name in names]
      stack = list_operation(*lists)
      return [dict(zip(names, values)) for values in stack]
    else:
      return previous_outputs

  def _join_lists_or_dicts(self, outputs, unit):
    """Joins many lists or dicts of outputs into a single list or dict.

    This function also validates that the outputs are correct for the given
    Pipeline.

    If `outputs` is a list of lists, the lists are concated and the type of
    each object must match `unit.output_type`.

    If `output` is a list of dicts (mapping string names to lists), each
    key has its lists concated across all the dicts. The keys and types
    are validated against `unit.output_type`.

    Args:
      outputs: A list of lists, or list of dicts which map string names to
          lists.
      unit: A Pipeline which every output in `outputs` will be validated
          against. `unit` must produce the outputs it says it will produce.

    Returns:
      If `outputs` is a list of lists, a single list of outputs.
      If `outputs` is a list of dicts, a single dictionary mapping string names
      to lists of outputs.

    Raises:
      InvalidTransformOutputException: If anything in `outputs` does not match
      the type signature given by `unit.output_type`.
    """
    if not outputs:
      return []
    if isinstance(unit.output_type, dict):
      concated = dict([(key, list()) for key in unit.output_type.keys()])
      for d in outputs:
        if not isinstance(d, dict):
          raise InvalidTransformOutputException(
              'Expected dictionary output for %s with output type %s but '
              'instead got type %s' % (unit, unit.output_type, type(d)))
        if set(d.keys()) != set(unit.output_type.keys()):
          raise InvalidTransformOutputException(
              'Got dictionary output with incorrect keys for %s. Got %s. '
              'Expected %s' % (unit, d.keys(), unit.output_type.keys()))
        for k, val in d.items():
          if not isinstance(val, list):
            raise InvalidTransformOutputException(
                'DagOutput from %s for key %s is not a list.' % (unit, k))
          if not _all_are_type(val, unit.output_type[k]):
            raise InvalidTransformOutputException(
                'Some outputs from %s for key %s are not of expected type %s. '
                'Got types %s' % (unit, k, unit.output_type[k],
                                  [type(inst) for inst in val]))
          concated[k] += val
    else:
      concated = []
      for l in outputs:
        if not isinstance(l, list):
          raise InvalidTransformOutputException(
              'Expected list output for %s with outpu type %s but instead got '
              'type %s' % (unit, unit.output_type, type(l)))
        if not _all_are_type(l, unit.output_type):
          raise InvalidTransformOutputException(
              'Some outputs from %s are not of expected type %s. Got types %s'
              % (unit, unit.output_type, [type(inst) for inst in l]))
        concated += l
    return concated
