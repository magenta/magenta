
<img src="magenta-logo-bg.png" height="75">

[![Build Status](https://github.com/magenta/magenta/workflows/build/badge.svg)](https://github.com/magenta/magenta/actions?query=workflow%3Abuild)
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

Take a look at our [colab notebooks](https://magenta.tensorflow.org/demos/colab/) for various models, including one on [getting started](https://colab.research.google.com/notebooks/magenta/hello_magenta/hello_magenta.ipynb).
[Magenta.js](https://github.com/tensorflow/magenta-js) is a also a good resource for models and [demos](https://magenta.tensorflow.org/demos/web/) that run in the browser.
This and more, including [blog posts](https://magenta.tensorflow.org/blog) and [Ableton Live plugins](https://magenta.tensorflow.org/demos/native/), can be found at [https://magenta.tensorflow.org](https://magenta.tensorflow.org).

## Magenta Repo

* [Installation](#installation)
* [Using Magenta](#using-magenta)
* [Development Environment (Advanced)](#development-environment)

## Installation

Magenta maintains a [pip package](https://pypi.python.org/pypi/magenta) for easy
installation. We recommend using Anaconda to install it, but it can work in any
standard Python environment. We support Python 3 (>= 3.5). These instructions
will assume you are using Anaconda.

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

**NOTE**: In order to install the `rtmidi` package that we depend on, you may need to install headers for some sound libraries. On Ubuntu Linux, this command should install the necessary packages:

```bash
sudo apt-get install build-essential libasound2-dev libjack-dev portaudio19-dev
```
On Fedora Linux, use
```bash
sudo dnf group install "C Development Tools and Libraries"
sudo dnf install SAASound-devel jack-audio-connection-kit-devel portaudio-devel
```


The Magenta libraries are now available for use within Python programs and
Jupyter notebooks, and the Magenta scripts are installed in your path!

## Using Magenta

You can now train our various models and use them to generate music, audio, and images. You can
find instructions for each of the models by exploring the [models directory](magenta/models).

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
twine upload dist/magenta-N.N.N-py2.py3-none-any.whl
```
