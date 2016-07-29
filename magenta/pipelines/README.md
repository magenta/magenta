# Data processing in Magenta

A pipeline is a data processing unit that transforms input data types to output data types.

Files:

* [pipeline.py](https://github.com/tensorflow/magenta/blob/master/magenta/pipelines/pipeline.py) defines the Pipeline abstract class and utility functions for running a Pipeline instance.
* [pipelines_common.py](https://github.com/tensorflow/magenta/blob/master/magenta/pipelines/pipelines_common.py) contains some Pipeline implementations that convert to common data types, like QuantizedSequence and MonophonicMelody.

All pipelines implement the abstract class Pipeline. Each Pipeline defines what its input and output look like. A pipeline can take as input a single object or dictionary mapping names to inputs. The output is given as a list of objects or a dictionary mapping names to lists of outputs. This allows the pipeline to output multiple items from a single input.

Pipeline has two methods:

* [transform(input_object)](https://github.com/tensorflow/magenta/blob/master/magenta/pipelines/pipeline.py#L81) converts a single input to one or many outputs.
* [get_stats()](https://github.com/tensorflow/magenta/blob/master/magenta/pipelines/pipeline.py#L96) returns statistics about each call to transform.

For example,

```python
class MyPipeline(Pipeline):
  ...

print MyPipeline.input_type
> MyType1

print MyPipeline.output_type
> MyType2

my_input = MyType1(1, 2, 3)
outputs = MyPipeline.transform(my_input)
print outputs
> [MyType2(1), MyType2(2), MyType2(3)]

print MyPipeline.get_stats()
> {'how_many_ints': Counter(3), 'sum_of_ints': Counter(6)}
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

print MyPipeline.get_stats()
> {'how_many_ints': Counter(3), 'sum_of_ints': Counter(6), 'property_2_digits': Counter(4)}
```

If the output is a dictionary, the lengths of the output lists do not need to be the same. Also, this example should not imply that a Pipeline which takes a dictionary input must produce a dictionary output, or vice versa. Pipelines do need to produce the type signature given by `output_type`.

Declaring `input_type` and `output_type` allows pipelines to be strung together inside meta-pipelines. So if there a pipeline that converts TypeA to TypeB, and you need TypeA to TypeC, then you only need to create a TypeB to TypeC pipeline. The TypeA to TypeC pipeline just feeds TypeA-to-TypeB into TypeB-to-TypeC.

A pipeline can be run over a dataset using `run_pipeline_serial`, or `load_pipeline`. `run_pipeline_serial` saves the output to disk, while load_pipeline keeps the output in memory. Only pipelines that output protocol buffers can be used in `run_pipeline_serial` since the outputs are saved to TFRecord. If the pipeline's `output_type` is a dictionary, the keys are used as dataset names.

Functions are also provided for iteration over input data. `file_iterator` iterates over files in a directory, returning the raw bytes. `tf_record_iterator` iterates over TFRecords, returning protocol buffers.

## DAGPipeline
_Connecting pipelines together._

`Pipeline` transforms A to B - input data to output data. But almost always it is cleaner to decompose this mapping into smaller pipelines, each with their own output representations. The recommended way to do this is to make a third `Pipeline` that runs the first two inside it. Magenta provides [DAGPipeline](https://github.com/tensorflow/magenta/blob/master/magenta/pipelines/dag_pipeline.py) - a `Pipeline` which takes a directed asyclic graph, or DAG, of `Pipeline` objects and runs it.

Lets take a look at a real example. Magenta has `Quantizer` and `MonophonicMelodyExtractor` (defined [here](https://github.com/tensorflow/magenta/blob/master/magenta/pipelines/pipelines_common.py)). `Quantizer` takes note data in seconds and snaps, or quantizes, everything to a discrete grid of timesteps. It maps `NoteSequence` protocol buffers to [QuantizedSequence](https://github.com/tensorflow/magenta/blob/master/magenta/lib/sequences_lib.py) objects. `MonophonicMelodyExtractor` maps those `QuantizedSequence` objects to [MonophonicMelody](https://github.com/tensorflow/magenta/blob/master/magenta/lib/melodies_lib.py) objects. Finally, we want to partition the output into a training and test set. `MonophonicMelody` objects are fed into `RandomPartition`, yet another `Pipeline` which outputs a dictionary of two lists: training output and test output.

All of this is strung together in a `DAGPipeline` (code is [here](https://github.com/tensorflow/magenta/blob/master/magenta/models/shared/melody_rnn_create_dataset.py)). First each of the pipelines are intantiated with parameters:

```python
quantizer = pipelines_common.Quantizer(steps_per_beat=4)
melody_extractor = pipelines_common.MonophonicMelodyExtractor(
    min_bars=7, min_unique_pitches=5,
    gap_bars=1.0, ignore_polyphonic_notes=False)
encoder_pipeline = EncoderPipeline(melody_encoder_decoder)
partitioner = pipelines_common.RandomPartition(
    tf.train.SequenceExample,
    ['eval_melodies', 'training_melodies'],
    [FLAGS.eval_ratio])
```
Next, the DAG is defined. The DAG is encoded with a Python dictionary that maps each `Pipeline` instance to run to the pipelines it depends on. More on this below.

```python
dag = {quantizer: Input(music_pb2.NoteSequence),
       melody_extractor: quantizer,
       encoder_pipeline: melody_extractor,
       partitioner: encoder_pipeline,
       Output(): partitioner}
```

Finally, the composite pipeline is created with a single line of code:
```python
composite_pipeline = DAGPipeline(dag)
```

## DAG Specification

`DAGPipeline` takes a single argument: the DAG encoded as a Python dictionary. The DAG specifies how data will flow via connections between pipelines.

Each (key, value) pair is in this form, ```destination: dependency```.

Remember that the `Pipeline` object defines `input_type` and `output_type` which can be Python classes or dictionaries mapping string names to Python classes. Lets call these the _type signature_ of the pipeline's input and output data. Both the destination and dependency have type signatures, and the rule for the DAG is that every (destination, dependency) has the same type signature.

Specifically, if the destination is a `Pipeline`, then `destination.input_type` must have the same type signature as the dependency.

___Break down of dependency___

The dependency tells DAGPipeline what Pipeline's need to be run before running the destination. Like type signatures, a dependency can be a single object or a dictionary mapping string names to objects. In this case the objects are `Pipeline` instances (there is also one other allowed type, but more on that below), and not classes.

___Input and outputs___

Finally, we need to tell DAGPipeline where its inputs go, and which pipelines produce its outputs. This is done with `Input` and `Output` objects. `Input` is given the input type that DAGPipeline will take, like `Input(str)`. `Output` is given a string name, like `Output('some_output')`. `DAGPipeline` always outputs a dictionary, and each `Output` in the DAG produces another name, output pair in `DAGPipeline`'s output. Currently, only 1 input is supported.

An example:

```python
print (ToPipeline.output_type, ToPipeline.input_type)
> (OutputType, IntermediateType)

print (FromPipeline.output_type, FromPipeline.input_type)
> (IntermediateType, InputType)

to_pipe = ToPipeline()
from_pipe = FromPipeline()
dag = {from_pipe: Input(InputType),
       to_pipe: from_pipe,
       Output('my_output'): to_pipe}

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
dag = {from_pipe: Input(InputType),
       to_pipe: from_pipe,
       Output('my_output'): to_pipe}

dag_pipe = DAGPipeline(dag)
print dag_pipe.transform(InputType())
> {'my_output': [<__main__.OutputType object>]}
```

Multiple outputs can be created by connecting a `Pipeline` with dictionary output to `Output()` without a name.

```python
print (MyPipeline.output_type, MyPipeline.input_type)
> ({'output_1': Type1, 'output_2': Type2}, InputType)

my_pipeline = MyPipeline()
dag = {my_pipeline: Input(InputType),
       Output(): my_pipeline}

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
dag = {first_pipe: Input(InputType),
       second_pipe: first_pipe['type_1'],
       third_pipe: first_pipe['type_2'],
       last_pipe: {'type_x': second_pipe, 'type_y': third_pipe},
       Output('my_output'): last_pipe}

dag_pipe = DAGPipeline(dag)
print dag_pipe.transform(InputType())
> {'my_output': [<__main__.OutputType object>]}
```

List index syntax and dictionary dependencies can also be combined.

```python
print (FirstPipeline.output_type, FirstPipeline.input_type)
> ({'type_1': TypeA, 'type_2': TypeB}, InputType)

print (LastPipeline.output_type, LastPipeline.input_type)
> (OutputType, {'type_x': TypeB, 'type_y': TypeA})

first_pipe = FirstPipeline()
last_pipe = LastPipeline()
dag = {first_pipe: Input(InputType),
       last_pipe: {'type_x': first_pipe['type_2'], 'type_y': first_pipe['type_1']},
       Output('my_output'): last_pipe}

dag_pipe = DAGPipeline(dag)
print dag_pipe.transform(InputType())
> {'my_output': [<__main__.OutputType object>]}
```

Not every pipeline output needs to be connected to something, as long as at least one output is used.


## DAGPipeline Exceptions

### InvalidDAGException

Thrown when the DAG dictionary is not well formatted. This can be because a `destination: dependency` pair is not `Pipeline: Pipeline` or `Pipeline: {'name_1': Pipeline, ...}`. It is also thrown when Input is given as a destination, or Output is given as a dependency.

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

Thrown when a Pipeline does not feed into anyhing, or there is a directed cycle.

BadTopologyException is thrown in these examples:
```python
pipe1 = MyPipeline1()
pipe2 = MyPipeline2()
dag = {Output('output'): Input(MyType),
       pipe2: pipe1,
       pipe1: pipe2}
DAGPipeline(dag)
```

```python
pipe1 = MyPipeline1()
pipe2 = MyPipeline2()
pipe3 = MyPipeline3()
dag = {pipe1: Input(pipe1.input_type),
       pipe2: {'a': pipe1, 'b': pipe3},
       pipe3: pipe2,
       Output(): {'pipe2': pipe2, 'pipe3': pipe3}}
DAGPipeline(dag)
```

### NotConnectedException

Thrown when a Pipeline used in a dependency has nothing feeding into it.

NotConnectedException is thrown in these examples:
```python
pipe1 = MyPipeline1()
pipe2 = MyPipeline2()
dag = {pipe1: Input(pipe1.input_type),
       Output('output'): pipe2}
DAGPipeline(dag)
```

```python
pipe1 = MyPipeline1()
pipe2 = MyPipeline2()
dag = {pipe1: Input(pipe1.input_type),
       Output(): {'pipe1': pipe1, 'pipe2': pipe2}}
DAGPipeline(dag)
```

### BadInputOrOutputException

Thrown when there is no Inputs or more than one Input with different types, or there is no Output.

BadInputOrOutputException is thrown in these examples:
```python
pipe1 = MyPipeline1()
dag = {pipe1: Input(pipe1.input_type)}
DAGPipeline(dag)

dag = {Output('output'): pipe1}
DAGPipeline(dag)

pipe2 = MyPipeline2()
dag = {pipe1: Input(Type1),
       pipe2: Input(Type2),
       Output(): {'pipe1': pipe1, 'pipe2': pipe2}}
DAGPipeline(dag)
```

Having multiple `Input` instances with the same type is allowed.

This example will not throw an exception:
```python
pipeA = MyPipelineA()
pipeB = MyPipelineB()
dag = {pipeA: Input(MyType),
       pipeB: Input(MyType),
       Output(): {'pipeA': pipeA, 'pipeB': pipeB}}
DAGPipeline(dag)
```

### InvalidStatisticsException

Thrown when a Pipeline in the DAG returns a value from its `get_stats` method which is not a Statistic instance.

### InvalidDictionaryOutput

Thrown when Output() is used without a dictionary dependency, or Output(name) is given with a name and with dictionary dependency.

InvalidDictionaryOutput is thrown in these examples:
```python
pipe1 = MyPipeline1()
pipe2 = MyPipeline2()
dag = {Output('output'): {'pipe1': pipe1, 'pipe2': pipe2}, ...}
DAGPipeline(dag)
```

```python
print MyPipeline.output_type
> MyType

pipe = MyPipeline()
dag = {Output(): pipe, ...}
DAGPipeline(dag)
```

### InvalidTransformOutputException

Thrown when a Pipeline in the DAG does not output the type(s) it gave in its `output_type`.

## Statistics

Statistics are great for collecting information about a dataset, and inspecting why a dataset created by a `Pipeline` turned out the way it did. Stats collected by `Pipeline`s need to be able to do two things: merge statistics together, and print out statistics.

A [Statistic](https://github.com/tensorflow/magenta/blob/master/magenta/pipelines/statistics.py) abstract class is provided. Each `Statistic` needs to define a `merge_from` method which combines data from another statistic of the same type, and a `pretty_print` method.

Two statistics are defined: `Counter` and `Histogram`.

`Counter` keeps a count as the name suggests, and has an increment function.

```python
count = Counter()
count.increment()
count.increment(5)
print count.pretty_print('my_counter')
> my_counter: 6
```

`Histogram` keeps counts for ranges of values, and also has an increment function.

```python
histogram = Histogram([0, 1, 2, 3])
histogram.increment(0.1)
histogram.increment(1.2)
histogram.increment(1.9)
histogram.increment(2.5)
print histogram.pretty_print('my_histogram')
> my_histogram:
>   [0,1): 1
>   [1,2): 2
>   [2,3): 1
```
