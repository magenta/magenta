<img src="http://magenta.tensorflow.org/assets/magenta-logo.png" height="75">

**Magenta** is a project from the [Google Brain team](https://research.google.com/teams/brain/)
that asks: Can we use machine learning to create compelling art and music? If
so, how? If not, why not?  We’ll use [TensorFlow](https://www.tensorflow.org),
and we’ll release our models and tools in open source on this GitHub. We’ll also
post demos, tutorial blog postings, and technical papers. Soon we’ll begin
accepting code contributions from the community at large. If you’d like to keep
up on Magenta as it grows, you can read our [blog](http://magenta.tensorflow.org) and or join our
[discussion group](http://groups.google.com/a/tensorflow.org/forum/#!forum/magenta-discuss).

## Installation

### Docker
The easiest way to get started with Magenta is to use our Docker container.
First, [install Docker](https://docs.docker.com/engine/installation/). Next, run
this command:

```docker run -it -p 6006:6006 -v /tmp/magenta:/magenta-data tensorflow/magenta```

This will start a shell in a directory with all Magenta components compiled and
ready to run. It will also map port 6006 of the host machine to the container so
you can view TensorBoard servers that run within the container.

This also maps the directory ```/tmp/magenta``` on the host machine to
```/magenta-data``` within the Docker session. Windows users can change 
```/tmp/magenta``` to a path such as ```C:/magenta```, and Mac and Linux users 
can use a path relative to their home folder such as ```~/magenta```. **WARNING**: 
only data saved in ```/magenta-data``` will persist across Docker sessions.

The Docker image also includes several pre-trained models in
```/magenta/models```. For example, to generate some MIDI files using the
[Lookback RNN](magenta/models/lookback_rnn), run this command:

```
bazel run //magenta/models/lookback_rnn:lookback_rnn_generate -- \
--bundle_file=/magenta-models/lookback_rnn.mag \
--output_dir=/magenta-data/lookback_rnn/generated \
--num_outputs=10 \
--num_steps=128 \
--primer_melody="[60]"
```

**NOTE**: Verify that the ```--output_dir``` path matches the path you 
mapped as your shared folder when running the ```docker run``` command. This
example command presupposes that you are using ```/magenta-data``` as your 
shared folder from the example ```docker run``` command above.

One downside to the Docker container is that it is isolated from the host. If
you want to listen to a generated MIDI file, you'll need to copy it to the host
machine. Similarly, because our
[MIDI instrument interface](magenta/interfaces/midi) requires access to the host
MIDI port, it will not work within the Docker container. You'll need to use the
full Development Environment.

Note: Our Docker image is also available at ```gcr.io/tensorflow/magenta```.

### Development Environment
If you want to develop on Magenta, use our
[MIDI instrument interface](magenta/interfaces/midi) or preview MIDI files
without copying them out out of the Docker environment, you'll need to set up
the full Development Environment.

The installation has three components. You are going to need Bazel to build packages, TensorFlow to run models, and an up-to-date version of this repository.

First, clone this repository:

```git clone https://github.com/tensorflow/magenta.git```

Next, [install Bazel](http://www.bazel.io/docs/install.html). We recommend the
latest version, currently 0.3.1.

Finally,
[install TensorFlow](https://www.tensorflow.org/versions/master/get_started/os_setup.html).
We require version 0.10 or later.

Also, verify that your environment uses Python 2.7. We do aim to support
Python 3 eventually, but it is currently experimental.

After that's done, run the tests with this command:

```bazel test //magenta/...```

## Generating MIDI

You can now create your own melodies with TensorFlow using one of our models:

**[Basic RNN](magenta/models/basic_rnn)**: A simple recurrent neural network for predicting melodies.

**[Lookback RNN](magenta/models/lookback_rnn)**: A recurrent neural network for predicting melodies that uses custom inputs and labels.

**[Attention RNN](magenta/models/attention_rnn)**: A recurrent neural network for predicting melodies that uses attention.

## Using a MIDI Instrument

After you've trained one of the models above, you can use our [MIDI interface](magenta/interfaces/midi) to play with it interactively.
