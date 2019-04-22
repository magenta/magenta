
<img src="magenta-logo-bg.png" height="75">

[![Build Status](https://travis-ci.org/tensorflow/magenta.svg?branch=master)](https://travis-ci.org/tensorflow/magenta)
 [![PyPI version](https://badge.fury.io/py/magenta.svg)](https://badge.fury.io/py/magenta)

**Magenta** is a research project exploring the role of machine learning
in the process of creating art and music.  Primarily this
involves developing new deep learning and reinforcement learning
algorithms for generating songs, images, drawings, and other materials. But it's also
an exploration in building smart tools and interfaces that allow
artists and musicians to extend (not replace!) their processes using
these models.  Magenta was started by some researchers and engineers
from the [Google Brain team](https://research.google.com/teams/brain/),
but many others have contributed significantly to the project. We use
[TensorFlow](https://www.tensorflow.org) and release our models and
tools in open source on this GitHub.  If you’d like to learn more
about Magenta, check out our [blog](https://magenta.tensorflow.org),
where we post technical details.  You can also join our [discussion
group](https://groups.google.com/a/tensorflow.org/forum/#!forum/magenta-discuss).

This is the home for our Python TensorFlow library. To use our models in the browser with [TensorFlow.js](https://js.tensorflow.org/), head to the [Magenta.js](https://github.com/tensorflow/magenta-js) repository.

## Getting Started

* [Installation](#installation)
* [Using Magenta](#using-magenta)
* [Playing a MIDI Instrument](#playing-a-midi-instrument)
* [Development Environment (Advanced)](#development-environment)

## Installation

Magenta maintains a [pip package](https://pypi.python.org/pypi/magenta) for easy
installation. We recommend using Anaconda to install it, but it can work in any
standard Python environment. We support both Python 2 (>= 2.7) and Python 3 (>= 3.5).
These instructions will assume you are using Anaconda.

Note that if you want to enable GPU support, you should follow the [GPU Installation](#gpu-installation) instructions below.

### Automated Install (w/ Anaconda)

If you are running Mac OS X or Ubuntu, you can try using our automated
installation script. Just paste the following command into your terminal.

```bash
curl https://raw.githubusercontent.com/tensorflow/magenta/master/magenta/tools/magenta-install.sh > /tmp/magenta-install.sh
bash /tmp/magenta-install.sh
```

After the script completes, open a new terminal window so the environment
variable changes take effect.

The Magenta libraries are now available for use within Python programs and
Jupyter notebooks, and the Magenta scripts are installed in your path!

Note that you will need to run `source activate magenta` to use Magenta every
time you open a new terminal window.

### Manual Install (w/o Anaconda)

If the automated script fails for any reason, or you'd prefer to install by
hand, do the following steps.

Install the Magenta pip package:

```bash
pip install magenta
```

**NOTE**: In order to install the `rtmidi` package that we depend on, you may need to install headers for some sound libraries. On Linux, this command should install the necessary packages:

```bash
sudo apt-get install build-essential libasound2-dev libjack-dev
```

The Magenta libraries are now available for use within Python programs and
Jupyter notebooks, and the Magenta scripts are installed in your path!

### GPU Installation

If you have a GPU installed and you want Magenta to use it, you will need to
follow the [Manual Install](#manual-install) instructions, but with a few
modifications.

First, make sure your system meets the [requirements to run tensorflow with GPU support](
https://www.tensorflow.org/install/install_linux#nvidia_requirements_to_run_tensorflow_with_gpu_support).

Next, follow the [Manual Install](#manual-install) instructions, but install the
`magenta-gpu` package instead of the `magenta` package:

```bash
pip install magenta-gpu
```

The only difference between the two packages is that `magenta-gpu` depends on
`tensorflow-gpu` instead of `tensorflow`.

Magenta should now have access to your GPU.

## Using Magenta

You can now train our various models and use them to generate music, audio, and images. You can
find instructions for each of the models by exploring the [models directory](magenta/models).

To get started, create your own melodies with TensorFlow using one of the various configurations of our [Melody RNN](magenta/models/melody_rnn) model; a recurrent neural network for predicting melodies.

## Playing a MIDI Instrument

After you've trained one of the models above, you can use our [MIDI interface](magenta/interfaces/midi) to play with it interactively.

We also have created several [demos](https://github.com/tensorflow/magenta-demos) that provide a UI for this interface, making it easier to use (e.g., the browser-based [AI Jam](https://github.com/tensorflow/magenta-demos/tree/master/ai-jam-js)).

## Development Environment
If you want to develop on Magenta, you'll need to set up the full Development Environment.

First, clone this repository:

```bash
git clone https://github.com/tensorflow/magenta.git
```

Next, install the dependencies by changing to the base directory and executing the setup command:

```bash
pip install -e .
```

You can now edit the files and run scripts by calling Python as usual. For example, this is how you would run the `melody_rnn_generate` script from the base directory:

```bash
python magenta/models/melody_rnn/melody_rnn_generate --config=...
```

You can also install the (potentially modified) package with:

```bash
pip install .
```

Before creating a pull request, please also test your changes with:

```bash
pip install pytest-pylint
pytest
```

## PIP Release

To build a new version for pip, bump the version and then run:

```bash
python setup.py test
python setup.py bdist_wheel --universal
python setup.py bdist_wheel --universal --gpu
twine upload dist/magenta-N.N.N-py2.py3-none-any.whl
twine upload dist/magenta_gpu-N.N.N-py2.py3-none-any.whl
```
