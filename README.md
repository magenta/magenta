<img src="magenta-logo-bg.png" height="75">

**Magenta** is a project from the [Google Brain team](https://research.google.com/teams/brain/)
that asks: Can we use machine learning to create compelling art and music? If
so, how? If not, why not?  We’ll use [TensorFlow](https://www.tensorflow.org),
and we’ll release our models and tools in open source on this GitHub. We’ll also
post demos, tutorial blog postings, and technical papers. If you’d like to keep
up on Magenta as it grows, you can read our [blog](http://magenta.tensorflow.org) and or join our
[discussion group](http://groups.google.com/a/tensorflow.org/forum/#!forum/magenta-discuss).

## Installation

### Python Pip

Magenta maintains a [pip package](https://pypi.python.org/pypi/magenta) for easy
installation. We recommend using Anaconda to install it, but it can work in any
standard Python 2.7 environment. These instructions will assume you are using
Anaconda.

#### Automated Install

If you are running Mac OS X or Ubuntu, you can try using our automated
installation script. Just paste the following command into your terminal.

```
curl https://raw.githubusercontent.com/tensorflow/magenta/master/magenta/tools/magenta-install.sh > /tmp/magenta-install.sh
bash /tmp/magenta-install.sh
```

After the script completes, open a new terminal window so the environment
variable changes take effect.

The Magenta libraries are now available for use within Python programs and
Jupyter notebooks, and the Magenta scripts are installed in your path!

Note that you will need to run `source activate magenta` to use Magenta every
time you open a new terminal window.

#### Manual Install

If the automated script fails for any reason, or you'd prefer to install by
hand, do the following steps.

First, download the
[Python 2.7 Miniconda installer](http://conda.pydata.org/miniconda.html) (you
can skip this step if you already have any variant of conda installed).

Next, create and activate a Magenta conda environment using Python 2.7 with
Jupyter notebook support:

```
conda create -n magenta python=2.7 jupyter
source activate magenta
```

Install the Magenta pip package:

```
pip install magenta
```

The Magenta libraries are now available for use within Python programs and
Jupyter notebooks, and the Magenta scripts are installed in your path!

Note that you will need to run `source activate magenta` to use Magenta every
time you open a new terminal window.

### Docker
Another way to try out Magenta is to use our Docker container.
First, [install Docker](https://docs.docker.com/engine/installation/). Next, run
this command:

```
docker run -it -p 6006:6006 -v /tmp/magenta:/magenta-data tensorflow/magenta
```

This will start a shell in a directory with all Magenta components compiled,
installed, and ready to run. It will also map port 6006 of the host machine to
the container so you can view TensorBoard servers that run within the container.

This also maps the directory `/tmp/magenta` on the host machine to
`/magenta-data` within the Docker session. Windows users can change
`/tmp/magenta` to a path such as `C:/magenta`, and Mac and Linux users
can use a path relative to their home folder such as `~/magenta`.
**WARNING**: only data saved in `/magenta-data` will persist across Docker
sessions.

The Docker image also includes several pre-trained models in
`/magenta/models`. For example, to generate some MIDI files using the
[Lookback Melody RNN](magenta/models/melody_rnn#lookback), run this command:

```
melody_rnn_generate \
  --config=lookback_rnn \
  --bundle_file=/magenta-models/lookback_rnn.mag \
  --output_dir=/magenta-data/lookback_rnn/generated \
  --num_outputs=10 \
  --num_steps=128 \
  --primer_melody="[60]"
```

**NOTE**: Verify that the `--output_dir` path matches the path you
mapped as your shared folder when running the `docker run` command. This
example command presupposes that you are using `/magenta-data` as your
shared folder from the example `docker run` command above.

One downside to the Docker container is that it is isolated from the host. If
you want to listen to a generated MIDI file, you'll need to copy it to the host
machine. Similarly, because our
[MIDI instrument interface](magenta/interfaces/midi) requires access to the host
MIDI port, it will not work within the Docker container. You'll need to use the
full Development Environment.

You may find at some point after installation that we have released a new version of Magenta and your Docker image is out of date. To update the image to the latest version, run:

```
docker pull tensorflow/magenta
```

Note: Our Docker image is also available at `gcr.io/tensorflow/magenta`.

## Generating MIDI

You can now create your own melodies with TensorFlow using one of the various configurations of our
[Melody RNN](magenta/models/melody_rnn) model; a recurrent neural network for predicting melodies.

## Using a MIDI Instrument

After you've trained one of the models above, you can use our [MIDI interface](magenta/interfaces/midi) to play with it interactively.

## Development Environment
If you want to develop on Magenta, you'll need to set up the full Development
Environment.

The installation has three components. You are going to need Bazel to build packages, TensorFlow to run models, and an up-to-date version of this repository.

First, clone this repository:

```
git clone https://github.com/tensorflow/magenta.git
```

Next, [install Bazel](https://bazel.build/docs/install.html). We require the
latest version, currently 0.4.5.

You will also need to install some required python dependencies. We recommend
using a conda environment and installing with pip:

```
pip install matplotlib scipy
```

Finally,
[install TensorFlow](https://www.tensorflow.org/get_started/os_setup.html).
To see what version of TensorFlow the code currently requires, check the
dependency listed in [setup.py](magenta/tools/pip/setup.py).

Also, verify that your environment uses Python 2.7. We do aim to support
Python 3 eventually, but it is currently experimental.

After that's done, run the tests with this command:

```
bazel test //magenta/...
```

To build and install the pip package from source, follow the
[pip build instructions](magenta/tools/pip#building-the-package). You can also
use our [build script](magenta/tools/build.sh).

If you want to build and run commands with Bazel, you'll need to run the package
that the build step generates. There are two ways to do this. The first option is
to look at the output of the build command to find the path to the generated file.
For example, if you want to build the melody_rnn_generate script:

```
$ bazel build //magenta/models/melody_rnn:melody_rnn_generate
INFO: Found 1 target...
Target //magenta/models/melody_rnn:melody_rnn_generate up-to-date:
  bazel-bin/magenta/models/melody_rnn/melody_rnn_generate

$ bazel-bin/magenta/models/melody_rnn/melody_rnn_generate --config=...
```

The other option is to use the `bazel run` command, which combines the two steps
above. Note that if you use `bazel run`, you'll need to add an extra `--` before
the command line arguments to differentiate between Bazel arguments and arguments
to the command.

```
$ bazel run //magenta/models/melody_rnn:melody_rnn_generate -- --config=...
```
