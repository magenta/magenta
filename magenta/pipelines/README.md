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

```
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
> {'how_many_ints': 3, 'sum_of_ints': 6}
```

An example where inputs and outputs are dictionaries,

```
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
> {'how_many_ints': 3, 'sum_of_ints': 6, 'property_2_digits': 4}
```

If the output is a dictionary, the lengths of the output lists do not need to be the same. Also, this example should not imply that a Pipeline which takes a dictionary input must produce a dictionary output, or vice versa. Pipelines do need to produce the type signature given by `output_type`.

Declaring `input_type` and `output_type` allows pipelines to be strung together inside meta-pipelines. So if there a pipeline that converts TypeA to TypeB, and you need TypeA to TypeC, then you only need to create a TypeB to TypeC pipeline. The TypeA to TypeC pipeline just feeds TypeA-to-TypeB into TypeB-to-TypeC.

A pipeline can be run over a dataset using `run_pipeline_serial`, or `load_pipeline`. `run_pipeline_serial` saves the output to disk, while load_pipeline keeps the output in memory. Only pipelines that output protocol buffers can be used in `run_pipeline_serial` since the outputs are saved to TFRecord. If the pipeline's `output_type` is a dictionary, the keys are used as dataset names.

Functions are also provided for iteration over input data. `file_iterator` iterates over files in a directory, returning the raw bytes. `tf_record_iterator` iterates over TFRecords, returning protocol buffers.