#!/bin/bash

# Fail on any error.
set -e
# Display commands to stderr.
set -x

# Ensure python 3.5 is used.
PY3_PATH="$(which python3.5)"

cd github/magenta

sudo apt-get -y install build-essential libasound2-dev libjack-dev
$PY3_PATH setup.py test --addopts="--pylint --disable-warnings"
