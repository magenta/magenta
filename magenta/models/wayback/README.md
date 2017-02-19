# Waybackpropagation through time

This tree contains code to train and sample from teacher-forcing models. The
code was developed to support ongoing research on audio synthesis on the
waveform level.

In particular, I explored "waybackpropagation through time": the use of
multi-scale recurrent connection patterns in combination with truncated
backpropagation to learn exponentially longer dependencies given a constant
memory budget.

## Training

To train a deep LSTM, run

```shell
blaze run -- train
  --data_dir /path/to/data
  --base_output_dir /tmp/stack
  --resume false
  --max_step_count 1000
  --max_examples 1
  --validation_interval 100
  --basename stack
  --hyperparameters '{
    sampling_frequency: 11025,
    bit_depth: 8,
    data_dim: 256,
    initial_learning_rate: 0.01,
    decay_rate: 0.1,
    decay_patience: 100,
    clip_norm: 1,
    layout: stack,
    cell: lstm,
    activation: elu,
    use_bn: True,
    vskip: True,
    batch_size: 100,
    segment_length: 100,
    chunk_size: 4,
    layer_sizes: [1000, 1000, 1000],
    io_sizes: [256],
    weight_decay: 1e-5
  }'
```

(The command above is formatted with newlines and indentation for clarity. For
execution in a shell it should be all one line or the newlines need to be
escaped.)

The `--data_dir` flag specifies the directory in which to look for data files.
The directory must have subdirectories named `train`, `valid` and `test`. The
training set is compiled by taking all files that match the `train/*.wav` glob
expression. The validation and test sets are constructed similarly.

Model checkpoints and hyperparameters will be saved in a subdirectory under
`--base_output_dir`. The name of the subdirectory is determined by `--basename`
and model hyperparameters.

`--resume` determined whether to resume from the latest checkpoint in the output
directory or start fresh.

`--max_step_count` limits the number of SGD steps taken. `None` means no limit.

`--max_examples` limits the number of training examples. This is useful for
model debugging purposes.

The `--validation_interval` setting controls how often (in SGD steps) to
evaluate the model on the validation set.

The `--hyperparameters` flag specifies the values of the model hyperparameters:

* The `sampling_frequency` is the desired sampling frequency for the waveform
  data. Any WAV files presented to the model will be resampled to this sampling
  frequency.

* The `bit_depth` is the desired amplitude resolution for the waveform data. The
  waveform amplitude will be discretized into `2 ** bit_depth` levels, and the
  model will treat the waveform as a sequence of categorical data.

* `data_dim` is the number of categories in the data. This is usually inferred
  to be `2 ** bit_depth`.

* `initial_learning_rate` specifies the initial learning rate for the SGD
  optimization process. If you set this too high, the optimization may not get
  anywhere. If you set it too low, optimization will progress too slowly. A
  learning rate on the order of `0.001` is usually fine.

* If the optimization does not make any progress (as measured by validation
  loss) for `decay_patience` steps, it will be reduced through multiplication
  by `decay_rate`. This makes the optimization process more robust to learning
  rate misspecification.

* Gradient clipping provides further protection against excessive parameter
  updates. Before taking an SGD step, the gradient is rescaled to have norm
  at most `clip_norm`.

* The `layout` controls the recurrent connection pattern; it can be either
  `stack` or `wayback`.

* The recurrent transition to use is determined by `cell`, which can be either
  `rnn`, `gru` or `lstm`.

* `activation` specifies the activation function to use inside the recurrent
  transition. This can be either `identity`, `elu` or `tanh`.

* `use_bn` controls whether to use recurrent batch normalization, which can
  help keep the activation and gradient dynamics in check for unbounded
  activation functions such as `identity` or `elu`.

* `vskip` enables vertical skip connections, such that the recurrent transition
  of each layer is conditioned on the state of all other layers.

* `batch_size` controls the size of SGD minibatches. Typical values are between
  50 and 100. Higher values make optimization easier but may result in
  overfitting.

* The model is trained by truncated backpropagation through time, on segments
  of length `segment_length`.

* To increase the opportunity for parallel computation, sequence data can be
  processed in chunks of `chunk_size` elements at a time. The probability
  distribution is still fully decomposed into stepwise conditional
  distributions; the chunk elements are modeled by a stateless autoregressive
  model conditioned on the recurrent state. Note that due to static unrolling,
  `segment_length` must be an integer multiple of `chunk_size`.

* The number of hidden units in the recurrent layers is given by `layer_sizes`,
  from bottom to top.

* The model can either interface with the data directly, or through
  fully-connected feed-forward networks on both the input and the
  output. `io_sizes` specifies the sizes of the layers in these nets. The sizes
  are shared between both the input and the output networks, but their
  parameters are independent.

* `weight_decay` indicates the coefficient for an L2 weight decay term in the
  loss function. This is especially helpful in conjunction with batch
  normalization, where the weights can sometimes grow large even though their
  magnitude is divided out.

To train a wayback model, run

```shell
blaze run -- train
  --data_dir /path/to/data
  --base_output_dir /tmp/models
  --resume false
  --max_step_count 1000
  --max_examples 1
  --validation_interval 100
  --basename wayback
  --hyperparameters '{
    sampling_frequency: 11025,
    bit_depth: 8,
    data_dim: 256,
    initial_learning_rate: 0.01,
    decay_rate: 0.1,
    decay_patience: 100,
    clip_norm: 1,
    layout: wayback,
    cell: gru,
    activation: elu,
    use_bn: True,
    vskip: True,
    batch_size: 100,
    segment_length: 1155,
    chunk_size: 3,
    layer_sizes: [1000, 1000, 1000],
    periods: [5, 7, 11],
    unroll_layer_count: 2,
    carry: False,
    io_sizes: [256],
    weight_decay: 1e-5
  }'
```

The wayback layout takes three additional hyperparameters:

* The `periods` determine how often each recurrent layer should update. In this
  case, the lowest layer will update every 3 steps (i.e. every chunk). The layer
  above it will update every 15 steps. The topmost layer will update every 105
  steps. The last element of `periods` specifies how many steps of the topmost
  layer to take before considering the cycle to be complete.

  Note that the product of the `periods` and `chunk_size` provides a lower bound
  on `segment_length`: it does not make sense to truncate backpropagation
  through time in such a way that some layer has not been updated at all.

* `unroll_layer_count` specifies the number of upper layers to unroll. Due to
  Tensorflow limitations it is not possible to stop gradient propagation inside
  dynamic loops. Partial static unrolling allows us to at least stop gradient
  propagation _between_ dynamic loops.

* If `carry` is set to `False`, the hidden states of lower layers are reset
  each time they complete a cycle. The new hidden state is a function of the
  state of the layer above it. This forces the model to make use of the extra
  capacity in the higher layers.

## Sampling

To sample from a trained model, run:

```shell
blaze run -- sample
  --model_ckpt /path/to/model.ckpt
  --model_hyperparameters /path/to/hyperparameters.yaml
  --base_output_path /path/to/output
  --temperature 0.001
  condition1.wav
  condition2.wav
  condition3.wav
```

* This loads the model checkpoint at the path specified by `--model_ckpt`.

* The `--model_hyperparameters` flag can be used to specify the path to a YAML
  file with model hyperparameters. It is important that the hyperparameters
  match those used during training of the model. For convenience, this flag
  usually does not need to be provided; by default the script uses the
  `hyperparameters.yaml` file that was written at training time to the same
  directory as the model checkpoint.

* All output paths are prefixed with `--base_output_path`. Include a trailing
  slash if the base output path refers to a directory.  By default, output files
  are written to the directory in which the model checkpoint resides.

* `--temperature` determines the softmax temperature. This is a positive
  parameter that can be used to control the entropy of the output distribution
  before sampling from it. At low temperatures the samples are more informed by
  the model, whereas at high temperatures the distribution is closer to being
  uniformly random.

Any additional arguments are taken to be paths to WAV files to condition on.
The sampling logic will take fragments from these files and run them through
the model before starting the generating process. The samples that are written
out will start with the conditioning fragment.
