# Data processing in Magenta

Magenta has a lot of different models which require different types of inputs. Some models train on melodies, some on raw audio or midi data. Being able to convert easily between these different data types is essential. We define a `Pipeline` which is a data processing module that transforms input data types to output data types. By connecting pipelines together, new data pipelines can be quickly built for new models.

Files:

* [pipeline.py](/magenta/pipelines/pipeline.py) defines the `Pipeline` abstract class and utility functions for running a `Pipeline` instance.
* [pipelines_common.py](/magenta/pipelines/pipelines_common.py) contains some `Pipeline` implementations that convert to common data types.
* [dag_pipeline.py](/magenta/pipelines/dag_pipeline.py) defines a `Pipeline` subclass `DAGPipeline` which connects arbitrary pipelines together inside it. These `Pipelines` can be connected into any directed acyclic graph (DAG).
* [statistics.py](/magenta/pipelines/statistics.py) defines the `Statistic` abstract class and implementations. Statistics are useful for reporting about data processing.
* [note_sequence_pipelines.py](/magenta/pipelines/note_sequence_pipelines.py) contains pipelines that operate on NoteSequences.
* [event_sequence_pipeline.py](/magenta/pipelines/event_sequence_pipeline.py), [chord_pipelines.py](/magenta/pipelines/chord_pipelines.py), [drum_pipelines.py](/magenta/pipelines/drum_pipelines.py), [melody_pipelines.py](/magenta/pipelines/melody_pipelines.py), and [lead_sheet_pipelines.py](/magenta/pipelines/lead_sheet_pipelines.py) define extractor pipelines for different types of musical event sequences.

## Pipeline

All pipelines implement the abstract class Pipeline. Each Pipeline defines what its input and output look like. A pipeline can take as input a single object or dictionary mapping names to inputs. The output is given as a list of objects or a dictionary mapping names to lists of outputs. This allows the pipeline to output multiple items from a single input.

Pipeline has two methods:

* `transform(input_object)` converts a single input to one or many outputs.
* `get_stats()` returns statistics (see `Statistic` objects below) about each call to transform.

And three important properties:

* `input_type` is the type signature that the pipeline expects for its inputs.
* `output_type` is the type signature of the pipeline's outputs.
* `name` is a unique string name of the pipeline. This is used as a namespace for `Statistic` objects the `Pipeline` produces.

For example,

```python
class MyPipeline(Pipeline):
  ...

print MyPipeline.input_type
> MyType1

print MyPipeline.output_type
> MyType2

print MyPipeline.name
> "MyPipeline"

my_input = MyType1(1, 2, 3)
outputs = MyPipeline.transform(my_input)
print outputs
> [MyType2(1), MyType2(2), MyType2(3)]

for stat in MyPipeline.get_stats():
  print str(stat)
> MyPipeline_how_many_ints: 3
> MyPipeline_sum_of_ints: 6
```

An example where inputs and outputs are dictionaries,

```python
class MyPipeline(Pipeline):
  ...

print MyPipeline.input_type
> {'property_1': MyType1, 'property_2': int}

print MyPipeline.output_type
> {'output_1': MyType2, 'output_2': str}

my_input = {'property_1': MyType1(1, 2, 3), 'property_2': 1007}
outputs = MyPipeline.transform(my_input)
print outputs
> {'output_1': [MyType2(1), MyType2(2), MyType2(3)], 'output_2': ['1007', '7001']}

for stat in MyPipeline.get_stats():
  print str(stat)
> MyPipeline_how_many_ints: 3
> MyPipeline_sum_of_ints: 6
> MyPipeline_property_2_digits: 4
```

If the output is a dictionary, the lengths of the output lists do not need to be the same. Also, this example should not imply that a Pipeline which takes a dictionary input must produce a dictionary output, or vice versa. Pipelines do need to produce the type signature given by `output_type`.

Declaring `input_type` and `output_type` allows pipelines to be strung together inside meta-pipelines. So if there a pipeline that converts TypeA to TypeB, and you need TypeA to TypeC, then you only need to create a TypeB to TypeC pipeline. The TypeA to TypeC pipeline just feeds TypeA-to-TypeB into TypeB-to-TypeC.

A pipeline can be run over a dataset using `run_pipeline_serial`, or `load_pipeline`. `run_pipeline_serial` saves the output to disk, while load_pipeline keeps the output in memory. Only pipelines that output protocol buffers can be used in `run_pipeline_serial` since the outputs are saved to TFRecord. If the pipeline's `output_type` is a dictionary, the keys are used as dataset names.

Functions are also provided for iteration over input data. `file_iterator` iterates over files in a directory, returning the raw bytes. `tf_record_iterator` iterates over TFRecords, returning protocol buffers.

Note that the pipeline name is prepended to the names of all the statistics in these examples. `Pipeline.get_stats` automatically prepends the pipeline name to the statistic name for each stat.

___Implementing Pipeline___

There are exactly two things a subclass of `Pipeline` is required to do:

1. Call `Pipeline.__init__` from its constructor passing in `input_type`, `output_type`, and `name`.
2. Implement the abstract method `transform`.

DO NOT override `get_stats`. To emit `Statistic` objects, call the private method `_set_stats` from the `transform` method. `_set_stats` will prepend the `Pipeline` name to all the `Statistic` names to avoid namespace conflicts, and write them to the private attribute `_stats`.

A full example:

```python
class FooExtractor(Pipeline):

  def __init__(self):
    super(FooExtractor, self).__init__(
        input_type=BarType,
        output_type=FooType,
        name='FooExtractor')

  def transform(self, bar_object):
    how_many_foo = Counter('how_many_foo')
    how_many_features = Counter('how_many_features')
    how_many_bar = Counter('how_many_bar', 1)

    features = extract_features(bar_object)
    foos = []
    for feature in features:
      if is_foo(feature):
        foos.append(make_foo(feature))
        how_many_foo.increment()
      how_many_features.increment()

    self._set_stats([how_many_foo, how_many_features, how_many_bar])
    return foos

foo_extractor = FooExtractor()
print FooExtractor.input_type
> BarType
print FooExtractor.output_type
> FooType
print FooExtractor.name
> "FooExtractor"

print foo_extractor.transform(BarType(5))
> [FooType(1), FooType(2)]

for stat in foo_extractor.get_stats():
  print str(stat)
> FooExtractor_how_many_foo: 2
> FooExtractor_how_many_features: 5
> FooExtractor_how_many_bar: 1
```

## DAGPipeline
___Connecting pipelines together___

`Pipeline` transforms A to B - input data to output data. But almost always it is cleaner to decompose this mapping into smaller pipelines, each with their own output representations. The recommended way to do this is to make a third `Pipeline` that runs the first two inside it. Magenta provides [DAGPipeline](/magenta/pipelines/dag_pipeline.py) - a `Pipeline` which takes a directed asyclic graph, or DAG, of `Pipeline` objects and runs it.

Lets take a look at a real example. Magenta has `Quantizer` (defined [here](/magenta/pipelines/note_sequence_pipelines.py)) and `MelodyExtractor` (defined [here](/magenta/pipelines/melody_pipelines.py)). `Quantizer` takes note data in seconds and snaps, or quantizes, everything to a discrete grid of timesteps. It maps `NoteSequence` protocol buffers to `NoteSequence` protos with quanitzed times. `MelodyExtractor` maps those quantized `NoteSequence` protos to [Melody](https://github.com/magenta/magenta/blob/master/note_seq/melodies_lib.py) objects. Finally, we want to partition the output into a training and test set. `Melody` objects are fed into `RandomPartition`, yet another `Pipeline` which outputs a dictionary of two lists: training output and test output.

All of this is strung together in a `DAGPipeline` (code is [here](/magenta/models/shared/melody_rnn_create_dataset.py)). First each of the pipelines are instantiated with parameters:

```python
quantizer = note_sequence_pipelines.Quantizer(steps_per_quarter=4)
melody_extractor = melody_pipelines.MelodyExtractor(
    min_bars=7, min_unique_pitches=5,
    gap_bars=1.0, ignore_polyphonic_notes=False)
encoder_pipeline = EncoderPipeline(config)
partitioner = pipelines_common.RandomPartition(
    tf.train.SequenceExample,
    ['eval_melodies', 'training_melodies'],
    [FLAGS.eval_ratio])
```
Next, the DAG is defined. The DAG is encoded with a Python dictionary that maps each `Pipeline` instance to run to the pipelines it depends on. More on this below.

```python
dag = {quantizer: DagInput(music_pb2.NoteSequence),
       melody_extractor: quantizer,
       encoder_pipeline: melody_extractor,
       partitioner: encoder_pipeline,
       DagOutput(): partitioner}
```

Finally, the composite pipeline is created with a single line of code:
```python
composite_pipeline = DAGPipeline(dag)
```

## Statistics

Statistics are great for collecting information about a dataset, and inspecting why a dataset created by a `Pipeline` turned out the way it did. Stats collected by `Pipeline`s need to be able to do three things: be copied, be merged together, and print out their information.

A [Statistic](/magenta/pipelines/statistics.py) abstract class is provided. Each `Statistic` needs to implement the `_merge_from` method which combines data from another statistic of the same type, the `copy` method, and the `_pretty_print` method.

Each `Statistic` also has a string name which identifies what is being measured. `Statistic`s with the same names get merged downstream.

Two statistic types are defined here: `Counter` and `Histogram`.

`Counter` keeps a count as the name suggests, and has an increment function.

```python
count = Counter('how_many_foo')
count.increment()
count.increment(5)
print str(count)
> how_many_foo: 6

count_2 = Counter('how_many_foo')
count_2.increment()
count.merge_from(count_2)
print str(count)
> how_many_foo: 7
```

`Histogram` keeps counts for ranges of values, and also has an increment function.

```python
histogram = Histogram('bar_distribution', [0, 1, 2, 3])
histogram.increment(0.0)
histogram.increment(1.2)
histogram.increment(1.9)
histogram.increment(2.5)
print str(histogram)
> bar_distribution:
>   [0,1): 1
>   [1,2): 2
>   [2,3): 1

histogram_2 = Histogram('bar_distribution', [0, 1, 2, 3])
histogram_2.increment(0.1)
histogram_2.increment(1.0, 3)
histogram.merge_from(histogram_2)
print str(histogram)
> bar_distribution:
>   [0,1): 2
>   [1,2): 5
>   [2,3): 1
```

When running `Pipeline.transform` many times, you will likely want to merge the outputs of `Pipeline.get_stats` into previous statistics. Furthermore, its possible for a `Pipeline` to produce many unmerged statistics. The `merge_statistics` method is provided to easily merge any statistics with the same names in a list.

```python
print my_pipeline.name
> "FooBarExtractor"

my_pipeline.transform(...)
stats = my_pipeline.get_stats()
for stat in stats:
  print str(stat)
> FooBarExtractor_how_many_foo: 1
> FooBarExtractor_how_many_foo: 1
> FooBarExtractor_how_many_bar: 1
> FooBarExtractor_how_many_foo: 0

merged = merge_statistics(stats)
for stat in merged:
  print str(stat)
> FooBarExtractor_how_many_foo: 2
> FooBarExtractor_how_many_bar: 1

my_pipeline.transform(...)
stats += my_pipeline.get_stats()
stats = merge_statistics(stats)
for stat in stats:
  print str(stat)
> FooBarExtractor_how_many_foo: 5
> FooBarExtractor_how_many_bar: 2
```

## DAG Specification

`DAGPipeline` takes a single argument: the DAG encoded as a Python dictionary. The DAG specifies how data will flow via connections between pipelines.

Each (key, value) pair is in this form, ```destination: dependency```.

Remember that the `Pipeline` object defines `input_type` and `output_type` which can be Python classes or dictionaries mapping string names to Python classes. Lets call these the _type signature_ of the pipeline's input and output data. Both the destination and dependency have type signatures, and the rule for the DAG is that every (destination, dependency) has the same type signature.

Specifically, if the destination is a `Pipeline`, then `destination.input_type` must have the same type signature as the dependency.

___Break down of dependency___

The dependency tells DAGPipeline what Pipeline's need to be run before running the destination. Like type signatures, a dependency can be a single object or a dictionary mapping string names to objects. In this case the objects are `Pipeline` instances (there is also one other allowed type, but more on that below), and not classes.

___Input and outputs___

Finally, we need to tell DAGPipeline where its inputs go, and which pipelines produce its outputs. This is done with `DagInput` and `DagOutput` objects. `DagInput` is given the input type that DAGPipeline will take, like `DagInput(str)`. `DagOutput` is given a string name, like `DagOutput('some_output')`. `DAGPipeline` always outputs a dictionary, and each `DagOutput` in the DAG produces another name, output pair in `DAGPipeline`'s output. Currently, only 1 input is supported.

___Usage examples___

A basic DAG:

```python
print (ToPipeline.output_type, ToPipeline.input_type)
> (OutputType, IntermediateType)

print (FromPipeline.output_type, FromPipeline.input_type)
> (IntermediateType, InputType)

to_pipe = ToPipeline()
from_pipe = FromPipeline()
dag = {from_pipe: DagInput(InputType),
       to_pipe: from_pipe,
       DagOutput('my_output'): to_pipe}

dag_pipe = DAGPipeline(dag)
print dag_pipe.transform(InputType())
> {'my_output': [<__main__.OutputType object>]}
```

Using dictionaries:

```python
print (ToPipeline.output_type, ToPipeline.input_type)
> (OutputType, {'intermediate_1': Type1, 'intermediate_2': Type2})

print (FromPipeline.output_type, FromPipeline.input_type)
> ({'intermediate_1': Type1, 'intermediate_2': Type2}, InputType)

to_pipe = ToPipeline()
from_pipe = FromPipeline()
dag = {from_pipe: DagInput(InputType),
       to_pipe: from_pipe,
       DagOutput('my_output'): to_pipe}

dag_pipe = DAGPipeline(dag)
print dag_pipe.transform(InputType())
> {'my_output': [<__main__.OutputType object>]}
```

Multiple outputs can be created by connecting a `Pipeline` with dictionary output to `DagOutput()` without a name.

```python
print (MyPipeline.output_type, MyPipeline.input_type)
> ({'output_1': Type1, 'output_2': Type2}, InputType)

my_pipeline = MyPipeline()
dag = {my_pipeline: DagInput(InputType),
       DagOutput(): my_pipeline}

dag_pipe = DAGPipeline(dag)
print dag_pipe.transform(InputType())
> {'output_1': [<__main__.Type1 object>], 'output_2': [<__main__.Type2 object>]}
```

What if you only want to use one output from a dictionary output, or connect multiple outputs to a dictionary input?

Heres how:

```python
print (FirstPipeline.output_type, FirstPipeline.input_type)
> ({'type_1': Type1, 'type_2': Type2}, InputType)

print (SecondPipeline.output_type, SecondPipeline.input_type)
> (TypeX, Type1)

print (ThirdPipeline.output_type, ThirdPipeline.input_type)
> (TypeY, Type2)

print (LastPipeline.output_type, LastPipeline.input_type)
> (OutputType, {'type_x': TypeX, 'type_y': TypeY})

first_pipe = FirstPipeline()
second_pipe = SecondPipeline()
third_pipe = ThirdPipeline()
last_pipe = LastPipeline()
dag = {first_pipe: DagInput(InputType),
       second_pipe: first_pipe['type_1'],
       third_pipe: first_pipe['type_2'],
       last_pipe: {'type_x': second_pipe, 'type_y': third_pipe},
       DagOutput('my_output'): last_pipe}

dag_pipe = DAGPipeline(dag)
print dag_pipe.transform(InputType())
> {'my_output': [<__main__.OutputType object>]}
```

List index syntax and dictionary dependencies can also be combined:

```python
print (FirstPipeline.output_type, FirstPipeline.input_type)
> ({'type_1': TypeA, 'type_2': TypeB}, InputType)

print (LastPipeline.output_type, LastPipeline.input_type)
> (OutputType, {'type_x': TypeB, 'type_y': TypeA})

first_pipe = FirstPipeline()
last_pipe = LastPipeline()
dag = {first_pipe: DagInput(InputType),
       last_pipe: {'type_x': first_pipe['type_2'], 'type_y': first_pipe['type_1']},
       DagOutput('my_output'): last_pipe}

dag_pipe = DAGPipeline(dag)
print dag_pipe.transform(InputType())
> {'my_output': [<__main__.OutputType object>]}
```

_Note:_ For pipelines which output more than one thing, not every pipeline output needs to be connected to something as long as at least one output is used.

DAGPipelines will collect the `Statistic` objects from its `Pipeline` instances:

```python
print pipeline_1.name
> 'Pipe1'
print pipeline_2.name
> 'Pipe2'

dag = {pipeline_1: DagInput(pipeline_1.input_type),
       pipeline_2: pipeline_1,
       DagOutput('output'): pipeline_2}
dag_pipe = DAGPipeline(dag)
print dag_pipe.name
> 'DAGPipeline'

dag_pipe.transform(Type1(5))
for stat in dag_pipe.get_stats():
  print str(stat)
> DAGPipeline_Pipe1_how_many_foo: 5
> DAGPipeline_Pipe1_how_many_bar: 2
> DAGPipeline_Pipe2_how_many_bar: 6
> DAGPipeline_Pipe2_foobar: 7
```

## DAGPipeline Exception

### InvalidDAGException

Thrown when the DAG dictionary is not well formatted. This can be because a `destination: dependency` pair is not in the form `Pipeline: Pipeline` or `Pipeline: {'name_1': Pipeline, ...}` (Note that Pipeline or Key objects both are allowed in the dependency). It is also thrown when `DagInput` is given as a destination, or `DagOutput` is given as a dependency.

### DuplicateNameException

Thrown when two `Pipeline` instances in the DAG have the same name. Pipeline names will be used as name spaces for the statistics they produce and we don't want any conflicts.

No Exception:
```python
print pipeline_1.name
> 'my_name'

print pipeline_2.name
> 'hello'

dag = {pipeline_1: ...,
       pipeline_2: ...}
DAGPipeline(dag)
```

DuplicateNameException thrown:
```python
print pipeline_1.name
> 'my_name'

print pipeline_2.name
> 'my_name'

dag = {pipeline_1: ...,
       pipeline_2: ...}
DAGPipeline(dag)
```

### TypeMismatchException

Thrown when destination type signature doesn't match dependency type signature.

TypeMismatchException is thrown in these examples:

```python
print DestPipeline.input_type
> Type1

print SrcPipeline.output_type
> Type2

dag = {DestPipeline(): SrcPipeline(), ...}
DAGPipeline(dag)
```

```python
print DestPipeline.input_type
> {'name_1': Type1}

print SrcPipeline.output_type
> {'name_1': Type2}

dag = {DestPipeline(): SrcPipeline(), ...}
DAGPipeline(dag)
```

```python
print DestPipeline.input_type
> {'name_1': MyType}

print SrcPipeline.output_type
> {'name_2': MyType}

dag = {DestPipeline(): SrcPipeline(), ...}
DAGPipeline(dag)
```

### BadTopologyException

Thrown when there is a directed cycle.

BadTopologyException is thrown in these examples:
```python
pipe1 = MyPipeline1()
pipe2 = MyPipeline2()
dag = {DagOutput('output'): DagInput(MyType),
       pipe2: pipe1,
       pipe1: pipe2}
DAGPipeline(dag)
```

```python
pipe1 = MyPipeline1()
pipe2 = MyPipeline2()
pipe3 = MyPipeline3()
dag = {pipe1: DagInput(pipe1.input_type),
       pipe2: {'a': pipe1, 'b': pipe3},
       pipe3: pipe2,
       DagOutput(): {'pipe2': pipe2, 'pipe3': pipe3}}
DAGPipeline(dag)
```

### NotConnectedException

Thrown when the DAG is disconnected somewhere. Either because a `Pipeline` used in a dependency has nothing feeding into it, or because a `Pipeline` given as a destination does not feed anywhere.

NotConnectedException is thrown in these examples:
```python
pipe1 = MyPipeline1()
pipe2 = MyPipeline2()
dag = {pipe1: DagInput(pipe1.input_type),
       DagOutput('output'): pipe2}
DAGPipeline(dag)
```

```python
pipe1 = MyPipeline1()
pipe2 = MyPipeline2()
dag = {pipe1: DagInput(pipe1.input_type),
       DagOutput(): {'pipe1': pipe1, 'pipe2': pipe2}}
DAGPipeline(dag)
```

```python
pipe1 = MyPipeline1()
pipe2 = MyPipeline2()
dag = {pipe1: DagInput(pipe1.input_type),
       pipe2: pipe1,
       DagOutput('pipe1'): pipe1}
DAGPipeline(dag)
```

### BadInputOrOutputException

Thrown when `DagInput` or `DagOutput` are not used in the graph correctly. Specifically when there are no `DagInput` objects, more than one `DagInput` with different types, or there is no `DagOutput` object.

BadInputOrOutputException is thrown in these examples:
```python
pipe1 = MyPipeline1()
dag = {pipe1: DagInput(pipe1.input_type)}
DAGPipeline(dag)

dag = {DagOutput('output'): pipe1}
DAGPipeline(dag)

pipe2 = MyPipeline2()
dag = {pipe1: DagInput(Type1),
       pipe2: DagInput(Type2),
       DagOutput(): {'pipe1': pipe1, 'pipe2': pipe2}}
DAGPipeline(dag)
```

Having multiple `DagInput` instances with the same type is allowed.

This example will not throw an Exception:
```python
pipeA = MyPipelineA()
pipeB = MyPipelineB()
dag = {pipeA: DagInput(MyType),
       pipeB: DagInput(MyType),
       DagOutput(): {'pipeA': pipeA, 'pipeB': pipeB}}
DAGPipeline(dag)
```

### InvalidStatisticsException

Thrown when a Pipeline in the DAG returns a value from its `get_stats` method which is not a list of Statistic instances.

Valid statistics:
```python
print my_pipeline.get_stats()
> [<Counter object>, <Histogram object>]
```

Invalid statistics:
```python
print my_pipeline_1.get_stats()
> ['hello', <Histogram object>]

print my_pipeline_2.get_stats()
> <Counter object>
```

### InvalidDictionaryOutput

Thrown when `DagOutput` and dictionaries are not used correctly. Specifically when `DagOutput()` is used without a dictionary dependency, or `DagOutput(name)` is used with a `name` and with a dictionary dependency.

InvalidDictionaryOutput is thrown in these examples:
```python
pipe1 = MyPipeline1()
pipe2 = MyPipeline2()
dag = {DagOutput('output'): {'pipe1': pipe1, 'pipe2': pipe2}, ...}
DAGPipeline(dag)
```

```python
print MyPipeline.output_type
> MyType

pipe = MyPipeline()
dag = {DagOutput(): pipe, ...}
DAGPipeline(dag)
```

### InvalidTransformOutputException

Thrown when a Pipeline in the DAG does not output the type(s) it promised to output in `output_type`.

This Pipeline would cause `InvalidTransformOutputException` to be thrown if it was passed into a DAGPipeline. It would also cause problems in general, and should be fixed no matter where it is used.
```python
print MyPipeline.output_type
> TypeA

print MyPipeline.transform(TypeA())
> [<__main__.TypeB object>]
```
